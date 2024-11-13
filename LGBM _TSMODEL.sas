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
	                copyvars=(&vf_byVars &vf_timeID &targetVar &vf_depVar);
	        run;

			/* Rename the column in score_next */
			proc cas;
				table.alterTable/ 
				name="&vf_tableOutPrefix..score_next",
				caslib="&vf_caslibOut",
				columns = {{name="&predictVar", rename="_inscalar_&predictVar"}};
			quit;

			/*Using TSMODEL to foward fill the lags of Y with predicted values */
	        proc tsmodel data=&vf_libOut.."&vf_tableOutPrefix..fxInData_forecasting"n
	            inscalar=&vf_libOut.."&vf_tableOutPrefix..score_next"n
	            outarray=&vf_libOut.."&vf_tableOutPrefix..fxInData_forecasting"n;
	            by &vf_byVars;
	            id &vf_timeID interval=&vf_timeIDInterval;
	            var &targetVar &predictVar &vf_depVar &vf_indepVars %intervalFeatureVarList;
	            inscalars _inscalar_&predictVar;
	
	            submit;
	                &predictVar.[&i.] =_inscalar_&predictVar;
	                %do j=1 %to &_lagYNumber;
		                %if (&i+1-&j > 0) %then %do;
		                    if _lagY&j[&i+1] = .  then do;
		                        _lagY&j[&i+1] = &predictVar[&i+1-&j];
		                    end;
		                %end; 
	                %end;
	            endsubmit;
	        run;
			
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