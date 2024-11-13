/*
Macro for adding features into input data
I: &vf_libIn.."&vf_inData"n
O: &vf_libOut..&vf_tableOutPrefix..&outTblName
*/
%macro fx_prepare_input(outTblName=, byVars=, trendVariable=, seasonalDummy=, 
		seasonalDummyInterval=, esmY=, lagXNumber=, lagYNumber=, holdoutSampleSize=, 
		holdoutSamplePercent=, criteria=, back=);
	%local filerootpath;
	%let filerootpath=&sas_root_location/misc/codegenscrpt/source/sas;
	%include "&filerootpath./vf_data_prep.sas";
	%let temp_work_location=%sysfunc(pathname(work));
	%include "&temp_work_location./vfDataPrepMacro.sas";
%mend;

/*
Main macro for forecasting using Light Gradient Boosting Model
*/
%macro lgbm_run;
	/* Protection against problematic _seasonDummy value */
	%if (not %symexist(_seasonDummy)) %then %let _seasonDummy=&vf_timeIDInterval;

	%if "&_seasonDummy" eq "" %then %let _seasonDummy=&vf_timeIDInterval;

	%if %sysfunc(INTTEST(&_seasonDummy)) eq 0 %then %do;
		%put Invalid seasonal dummy interval.
		Use &vf_timeIDInterval instead.;
		%let _seasonDummy=&vf_timeIDInterval;
	%end;

	/* Prepare input data with extracted features used for modeling */
	%fx_prepare_input(outTblName=fxInData, byVars=&vf_byVars, 
		trendVariable=&_trend, seasonalDummy=&_seasonDummy, 
		seasonalDummyInterval=&_seasonDummyInterval, esmY=FALSE, 
		lagXNumber=&_lagXNumber, lagYNumber=&_lagYNumber, 
		holdoutSampleSize=&_holdoutSampleSize, 
		holdoutSamplePercent=&_holdoutSamplePercent, criteria=RMSE, back=0);

	/* Dependent variable transformation if needed */
	%let targetVar=lgbmTargetVar;
	%let predictVar=P_&targetVar;

	data &vf_libOut.."&vf_tableOutPrefix..fxInData"n / SESSREF=&vf_session;
		set &vf_libOut.."&vf_tableOutPrefix..fxInData"n;
		&targetVar=&vf_depVar;

		%if %upcase(&_depTransform) eq LOG %then %do;

			if not missing(&vf_depVar) and &vf_depVar > 0 then 
				&targetVar=log(&vf_depVar);
			else call missing(&targetVar);
		%end;
	run;
	
    /*Drop previous table*/
    proc cas;
		table.droptable / name="&vf_tableOutPrefix..scored_lgb" caslib="&vf_caslibOut" quiet=true;
		table.droptable / name="&vf_tableOutPrefix..lgbmStore" caslib="&vf_caslibOut" quiet=true;
		table.droptable / name="&vf_outFor" caslib="&vf_caslibOut" quiet=true;
    quit;
    
	/* Train the Light Gradient Boosting Model */
	proc lightgradboost 
			data=&vf_libOut.."&vf_tableOutPrefix..fxInData"n(where=(_roleVar=1)) 
			validdata=&vf_libOut.."&vf_tableOutPrefix..fxInData"n(where=(_roleVar=2)) 
			seed=12345;
		input &vf_byVars /level=NOMINAL;

		%if "&vf_indepVars" ne "" %then %do;
			input &vf_indepVars /level=INTERVAL;
		%end;
		
		%if %intervalFeatureVarList ne %then %do;
			input %intervalFeatureVarList /level=INTERVAL;
		%end;

		%if %nominalFeatureVarList ne %then %do;
			input %nominalFeatureVarList /level=NOMINAL;
		%end;
		target &targetVar / level=interval;
		autotune maxtime=3600
		tuningparameters=(lasso(lb=0 ub=1 init=0));
		savestate rstore=&vf_libOut.."&vf_tableOutPrefix..lgbmStore"n;
	run;

	/* Score the model for the whole dataset when lagYNumber=0 */
	%if %eval(&_lagYNumber=0) %then %do;

		proc astore;
			score data=&vf_libOut.."&vf_tableOutPrefix..fxInData"n 
				out=&vf_libOut.."&vf_tableOutPrefix..scored_lgb"n
				rstore=&vf_libOut.."&vf_tableOutPrefix..lgbmStore"n 
				copyvars=(&vf_byVars &vf_timeID &targetVar &vf_depVar);
		run;

	%end;
	
	%if %eval(&_lagYNumber>0) %then %do;

        /*Create a list of variables that need to coalesce in fedsql*/
        %let coalesce_vars = coalesce(a.&predictVar,b.&predictVar) as &predictVar;
        %do i = 1 %to &_lagYNumber;
            %let coalesce_vars = &coalesce_vars, coalesce(a._lagY&i, b._lagY&i) as _lagY&i;
        %end;
        %put  coalesce_vars =  &coalesce_vars ;
        
        /* Initialize the macro variable to store the modified list */
		%let FeatureVarList = %intervalFeatureVarList;
		%put FeatureVarList = &FeatureVarList;
		%let FeatureVar_columns = ;
		/* Loop through each variable in the list */
		%let count = %sysfunc(countw(&FeatureVarList, %str( )));
		%do i = 1 %to &count;
		    %let var = %scan(&FeatureVarList, &i, %str( ));
		    
		    /* Check if the variable matches the pattern _lagY* */
		    %if %sysfunc(index(&var, _lagY)) = 0 %then %do;
		             /* If FeatureVar_columns is empty, set it to the current variable with prefix 'a.' */
		        %if %sysevalf(%superq(FeatureVar_columns)=,boolean) %then %let FeatureVar_columns = a.&var;
		        %else %let FeatureVar_columns = &FeatureVar_columns, a.&var;
		    %end;
		%end;
		
		%put FeatureVar_columns = &FeatureVar_columns;
				

	    /* Score both the training and validation data */
	    proc astore;
	        score data=&vf_libOut.."&vf_tableOutPrefix..fxInData"n(where=(_roleVar in (1, 2)))
	            out=&vf_libOut.."&vf_tableOutPrefix..scored_lgb_train"n
	            rstore=&vf_libOut.."&vf_tableOutPrefix..lgbmStore"n
	            copyvars=(&vf_byVars &vf_timeID &targetVar &vf_depVar &vf_indepVars %intervalFeatureVarList);
	    run;
	
	    /* Create forecasting dataset */
	    data &vf_libOut.."&vf_tableOutPrefix..fxInData_forecasting"n;
	        set &vf_libOut.."&vf_tableOutPrefix..fxInData"n(where=(_roleVar=0));
	        &predictVar = .;
	    run;
	
	    /* The number of iterations should be the steps of forecasting */
	    %do i=1 %to %eval(&vf_lead);
	        %let date_idx = %sysfunc(intnx(&vf_timeIDInterval, &vf_horizonStart, %eval(&i-1)));
			
	        proc astore;
	            score data=&vf_libOut.."&vf_tableOutPrefix..fxInData_forecasting"n(where=(&vf_timeID=&date_idx))
	                out=&vf_libOut.."&vf_tableOutPrefix..score_next"n
	                rstore=&vf_libOut.."&vf_tableOutPrefix..lgbmStore"n
	                copyvars=(&vf_byVars &vf_timeID);
	        run;

			/* Rename the column in score_next */
			proc cas;
				table.alterTable/ 
				name="&vf_tableOutPrefix..score_next",
				caslib="&vf_caslibOut",
				columns = {{name="&predictVar", rename="ypred"}};
			quit;
	
	        /*extend the scored table for filling the lags of Y*/
	        data &vf_libOut.."&vf_tableOutPrefix..score_next"n ;
	            set &vf_libOut.."&vf_tableOutPrefix..score_next"n;
                &predictVar = ypred;
                output;
                %do j=1 %to &_lagYNumber;
                    &vf_timeID = intnx("&vf_timeIDInterval", &vf_timeID, 1);
                    &predictVar = . ;
                    %do k=1 %to &_lagYNumber;
                        %if &k = &j %then %do;
                            _lagY&k = ypred;
                        %end;
                        %else %do;
                            _lagY&k = .;
                        %end;
                    %end;
                    output;
                %end;
                drop ypred;    
	        run;
	        
	       /*Using SQL to merge the tables and fill the missing values of future Y lags*/
			proc cas;
				/* Convert the JSON string to a CASL object */
				args = json2casl(symget('_backendArgs'));
				
				/* Print the args object for debugging */
				*print args;
				
				/* Access relevant properties from args */
				timeID_name = args.timeID.name;
				
				depVar_name = args.dataSpecification.dependentVar.name;
				indepVars_list = args.dataSpecification.independentVar;
		
				/* Initialize select_columns */
				indepVar_columns = "";
				

				/* Loop through independent variables and add to indepVar_columns */
				do indepVar over indepVars_list;
					indepVar_columns = indepVar_columns || ", a." || indepVar.name;
				end;
				
				/* Remove leading comma from indepVar_columns */
				indepVar_columns = substr(indepVar_columns, 3);

				


				/* Assign system macro variables to CASL variables using symget */
				vf_caslibOut = symget('vf_caslibOut');
				vf_tableOutPrefix = symget('vf_tableOutPrefix');
			
				/* Assign user-defined variables directly */
				targetVar = "&targetVar.";
				FeatureVar_columns = "&FeatureVar_columns.";
				coalesce_vars = "&coalesce_vars.";

				/* Print values for debugging */
				print vf_caslibOut;
				print vf_tableOutPrefix;
				print targetVar;
				print FeatureVar_columns;
				print coalesce_vars;


				if dim(args.dataSpecification.byVarsList) > 0 then do;
					byVars_list = args.dataSpecification.byVarsList;
					byVar_columns = "";
					/* Loop through by variables and add to byVar_columns */
					do byVar over byVars_list;
						byVar_columns = byVar_columns || ", a." || byVar;
					end;
					
					/* Remove leading comma from select_columns */
					byVar_columns = substr(byVar_columns, 3);
					/* Construct the byvarstatements for join conditions */
					byvarstatements = "";
						do byvar over byVars_list;
							byvarstatements = byvarstatements || "a." || byvar || "=b." || byvar ||  " and ";
						end;
					/* Remove trailing ' and ' from byvarstatements */
					byvarstatements = substr(byvarstatements, 1, length(byvarstatements) - 4);

					/* Construct the final SQL statement */
					sql_query = 
						'create table "' || vf_caslibOut || '"."' || vf_tableOutPrefix || '.fxInData_forecasting" {option replace=true} as select a.' || timeID_name || ', ' || byVar_columns || ', a.' || depVar_name || ', a.' || targetVar || 
						', ' || FeatureVar_columns || ', ' || indepVar_columns || ', ' || coalesce_vars || 
						' from "' || vf_caslibOut || '"."' || vf_tableOutPrefix || '.fxInData_forecasting" as a left join "' || vf_caslibOut || '"."' || vf_tableOutPrefix || '.score_next" as b on a.' || timeID_name || ' = b.' || timeID_name || ' and ' || byvarstatements ;
				end;

				else ;
				do;
					sql_query = 
						'create table "' || vf_caslibOut || '"."' || vf_tableOutPrefix || '.fxInData_forecasting" {option replace=true} as select a.' || timeID_name || ', a.' || depVar_name || ', a.' || targetVar || 
						', ' || FeatureVar_columns || ', ' || indepVar_columns || ', ' || coalesce_vars || 
						' from "' || vf_caslibOut || '"."' || vf_tableOutPrefix || '.fxInData_forecasting" as a left join "' || vf_caslibOut || '"."' || vf_tableOutPrefix || '.score_next" as b on a.' || timeID_name || ' = b.' || timeID_name ||;
				end;

				/* Print the constructed SQL query for debugging */
				print sql_query;

				/* Execute the SQL statement */
				fedsql.execdirect / query= sql_query;
			quit;
			
	    %end;
	
	    /* Concatenate both train table and forecasting table together */
	    data &vf_libOut.."&vf_tableOutPrefix..scored_lgb"n;
	        set &vf_libOut.."&vf_tableOutPrefix..scored_lgb_train"n &vf_libOut.."&vf_tableOutPrefix..fxInData_forecasting"n;
	    run;
        

		/* Drop the intermediate table when the iteration ends */
	    proc cas;
	        table.droptable / name="&vf_tableOutPrefix..score_next"
	            caslib="&vf_caslibOut"
	            quiet=true;
	        table.droptable / name="&vf_tableOutPrefix..scored_lgb_train"
	            caslib="&vf_caslibOut"
	            quiet=true;
			table.droptable / name="&vf_tableOutPrefix..fxInData_forecasting"
	            caslib="&vf_caslibOut"
	            quiet=true;
	    quit;
	%end;
	

	/* Prepare the required output tables */
	data &vf_libOut.."&vf_outFor"n;
		set &vf_libOut.."&vf_tableOutPrefix..scored_lgb"n;
		actual=&targetVar;
		predict=&predictVar;

		%if %upcase(&_depTransform) eq LOG %then %do;

			if not missing(actual) then actual=exp(actual);

			if not missing(predict) then predict=exp(predict);
		%end;

		%if "&vf_allowNegativeForecasts" eq "FALSE" %then %do;

			if not missing(predict) and predict < 0 then predict=0;
		%end;

		%if &targetVar ne &vf_depVar or &predictVar ne predict %then %do;
			drop &targetVar &predictVar;
		%end;
	run;

%mend;

/* Invoke the main macro */
%lgbm_run;
