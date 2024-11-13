/*
    macro for adding features into input data
        I: &vf_libIn.."&vf_inData"n
        O: &vf_libOut..&vf_tableOutPrefix..&outTblName
*/
%macro fx_prepare_input(outTblName=, byVars=, 
                        trendVariable=, seasonalDummy=, 
                        seasonalDummyInterval=,
                        esmY=, lagXNumber=, lagYNumber=, 
                        holdoutSampleSize=, holdoutSamplePercent=, 
                        criteria=, back=);

    %local filerootpath;
    %let filerootpath = &sas_root_location/misc/codegenscrpt/source/sas;
    %include "&filerootpath./vf_data_prep.sas";
    %let temp_work_location = %sysfunc(pathname(work));
    %include "&temp_work_location./vfDataPrepMacro.sas";

%mend;


/*
    main macro for forecasting using Gradient Boosting Model
*/
%macro gbm_run;

    /*protection against problematic _seasonDummy value*/
    %if (not %symexist(_seasonDummy)) %then 
        %let _seasonDummy=&vf_timeIDInterval;
    %if "&_seasonDummy" eq "" %then 
        %let _seasonDummy = &vf_timeIDInterval;
    %if %sysfunc(INTTEST( &_seasonDummy )) eq 0 %then %do;
        %put Invalid seasonal dummy interval. 
             Use &vf_timeIDInterval instead.;
        %let _seasonDummy = &vf_timeIDInterval;
    %end;
        
    /*parepare input data with extracted feature used for modeling*/
    %fx_prepare_input(outTblName=fxInData, byVars=&vf_byVars, 
                      trendVariable = &_trend,  
                      seasonalDummy = &_seasonDummy, 
                      seasonalDummyInterval = &_seasonDummyInterval,
                      esmY =FALSE, lagXNumber=&_lagXNumber, 
                      lagYNumber=&_lagYNumber, 
                      holdoutSampleSize=&_holdoutSampleSize, 
                      holdoutSamplePercent=&_holdoutSamplePercent, 
                      criteria=RMSE, back=0);
    
    /*Dependent variable transformation if needed*/
    %let targetVar=gbmTargetVar;
    %let predictVar=P_&targetVar;
    data &vf_libOut.."&vf_tableOutPrefix..fxInData"n / 
        SESSREF=&vf_session;
        set &vf_libOut.."&vf_tableOutPrefix..fxInData"n;
        &targetVar = &vf_depVar;
        %if %upcase(&_depTransform) eq LOG %then %do;
          if not missing(&vf_depVar) and &vf_depVar > 0 then 
             &targetVar = log(&vf_depVar);
          else call missing(&targetVar);
        %end;
    run;

    /*Drop previous table*/
    proc cas;
		table.droptable / name="&vf_tableOutPrefix..scored_gb" caslib="&vf_caslibOut" quiet=true;
		table.droptable / name="&vf_tableOutPrefix..gbmStore" caslib="&vf_caslibOut" quiet=true;
		table.droptable / name="&vf_outFor" caslib="&vf_caslibOut" quiet=true;
    quit;
    
    /*Train the Gradient Boosting Model*/
    proc gradboost data=&vf_libOut.."&vf_tableOutPrefix..fxInData"n 
                   seed=12345;
        /*We don't have id, partition in LightGBM 
        
        The ID statement lists one or more variables that are to be copied from the input data table 
        to the output data tables that are specified in the OUT= option in the OUTPUT statement and the 
        RSTORE= option in the SAVESTATE statement.

        */
        id &vf_byVars &vf_timeID;                          
        partition rolevar=_roleVar(TRAIN="1" VALIDATE="2" TEST="3");                 
        input &vf_byVars  /level=NOMINAL;
        %if "&vf_indepVars" ne "" %then %do;
          input &vf_indepVars  /level=INTERVAL;
        %end;
        %if %intervalFeatureVarList ne  %then %do;
          input %intervalFeatureVarList /level=INTERVAL;
        %end;
        %if %nominalFeatureVarList ne  %then %do;
          input %nominalFeatureVarList /level=NOMINAL;
        %end;
        target &targetVar / level=interval;
        /*The options in autotune of LightGBM is a little bit different*/
        autotune maxtime=3600
                 tuningparameters=( ntrees(lb=50 ub=500 init=50)) ;
        %if %eval(&_lagYNumber>0) %then %do;
          	savestate rstore=&vf_libOut.."&vf_tableOutPrefix..gbmStore"n;
        %end;
        %else %do;
           output out=&vf_libOut.."&vf_tableOutPrefix..scored_gb"n
                  copyvars=(&vf_byVars &vf_timeID &targetVar);
        %end;
    run;
    
    %if %eval(&_lagYNumber>0) %then %do;
    /* Score both the training and validation data */
	    proc astore;
	        score data=&vf_libOut.."&vf_tableOutPrefix..fxInData"n(where=(_roleVar in (1, 2)))
	            out=&vf_libOut.."&vf_tableOutPrefix..scored_gb_train"n
	            rstore=&vf_libOut.."&vf_tableOutPrefix..gbmStore"n
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
	                rstore=&vf_libOut.."&vf_tableOutPrefix..gbmStore"n
	                copyvars=(&vf_byVars &vf_timeID &targetVar &vf_depVar);
	        run;
	
			/* Rename the column in score_next */
			proc cas;
				table.alterTable/ 
				name="&vf_tableOutPrefix..score_next",
				caslib="&vf_caslibOut",
				columns = {{name="&predictVar", rename="_inscalar_&predictVar"}};
			quit;
	
	        proc tsmodel data=&vf_libOut.."&vf_tableOutPrefix..fxInData_forecasting"n 
	            inscalar=&vf_libOut.."&vf_tableOutPrefix..score_next"n
	            outarray=&vf_libOut.."&vf_tableOutPrefix..fxInData_forecasting"n ;
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
		    data &vf_libOut.."&vf_tableOutPrefix..scored_gb"n;
		        set &vf_libOut.."&vf_tableOutPrefix..scored_gb_train"n &vf_libOut.."&vf_tableOutPrefix..fxInData_forecasting"n;
		    run;

            /* Drop the intermediate table when the iteration ends */
            proc cas;
                table.droptable / name="&vf_tableOutPrefix..score_next"
                    caslib="&vf_caslibOut"
                    quiet=true;
                table.droptable / name="&vf_tableOutPrefix..scored_gb_train"
                    caslib="&vf_caslibOut"
                    quiet=true;
                table.droptable / name="&vf_tableOutPrefix..fxInData_forecasting"
                    caslib="&vf_caslibOut"
                    quiet=true;
            quit;
    %end;


    
    /*prepare the required output tables*/
    data &vf_libOut.."&vf_outFor"n;
       set &vf_libOut.."&vf_tableOutPrefix..scored_gb"n;
       actual = &targetVar;
       predict = &predictVar;
       %if %upcase(&_depTransform) eq LOG %then %do;
          if not missing(actual) then actual = exp(actual);
          if not missing(predict) then predict = exp(predict);
       %end;
       %if "&vf_allowNegativeForecasts" eq "FALSE" %then %do;
          if not missing(predict) and predict < 0 then predict = 0;
       %end;
       %if &targetVar ne &vf_depVar or &predictVar ne predict %then %do;
          drop &targetVar  &predictVar;
       %end;
    run;    

%mend;  

/*invoke the main macro */
%gbm_run;

