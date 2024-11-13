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
		
		/*Create extracted feature columns filled with missing values*/
        %do i = 1 %to 777;
            _FEATURE&i = .;
        %end;
	run;
	
    /*Drop previous table*/
    proc cas;
		table.droptable / name="&vf_tableOutPrefix..scored_lgb" caslib="&vf_caslibOut" quiet=true;
		table.droptable / name="&vf_tableOutPrefix..lgbmStore" caslib="&vf_caslibOut" quiet=true;
		table.droptable / name="&vf_outFor" caslib="&vf_caslibOut" quiet=true;
    quit;


	/*Create macro variable contains all feature names*/
	%let EXFEATURES=;
    %do i = 1 %to 777;
        %let EXFEATURES = &EXFEATURES _FEATURE&i;
    %end;

	proc tsmodel data=&vf_libOut.."&vf_tableOutPrefix..fxInData"n
		OUTARRAY=&vf_libOut.."&vf_tableOutPrefix..fxInData"n
		OUTSCALAR=&vf_libOut.."&vf_tableOutPrefix..outscalar"n
		OUTLOG=&vf_libOut.."&vf_tableOutPrefix..outlog"n
		/*nthreads=1*/
		LOGCONTROL=(error=keep warning=keep note=keep none=keep) PUTTOLOG=YES
		/*HERE WE NEED LEAD, BUT IN CUSTOM NODE, THE DATA IS ALREADY EXTENDED.*/
		OUTOBJ=(pylog=&vf_libOut.."&vf_tableOutPrefix..pylog"n (replace=YES));
		by &vf_byVars;
		id &vf_timeID interval=&vf_timeIDInterval;
		var &targetVar  &vf_depVar &vf_indepVars _roleVar %intervalFeatureVarList &EXFEATURES;
		OUTSCALAR runtime;
		REQUIRE extlang;
		PRINT outlog;
		SUBMIT;
		declare object py(PYTHON3);
		rc = py.Initialize();
		py_lead = &vf_lead;

		/* Dynamically add variables from EXFEATURES. _FEATURE1, _FEATURE2, etc. */
		%let i = 1;
		%let varname = %scan(&EXFEATURES, &i);
		%do %while(&varname ne );
		    rc = py.AddVariable(&varname, 'READONLY', 'FALSE');
		    %let i = %eval(&i + 1);
		    %let varname = %scan(&EXFEATURES, &i); /* Corrected this line */
		%end;

		/* Specify shared variables */
		rc = py.AddVariable(py_lead, "alias", "PY_LEAD");
		rc = py.AddVariable(&targetVar, "alias", "VF_TARGET");
		rc = py.AddVariable(&vf_timeID, "alias", "VF_ID");
		rc=py.PushCodeLine('import numpy as np');
		rc=py.PushCodeLine('import pandas as pd');
		rc=py.PushCodeLine("from statsmodels.tsa.holtwinters import ExponentialSmoothing");
		rc=py.PushCodeLine('from tsfresh import extract_features, extract_relevant_features, select_features');
		rc=py.PushCodeLine('from tsfresh.utilities.dataframe_functions import roll_time_series,make_forecasting_frame');
		rc=py.PushCodeLine('from tsfresh.feature_extraction import MinimalFCParameters,EfficientFCParameters');
		rc=py.PushCodeLine('from tsfresh.utilities.dataframe_functions import impute');
		rc=py.PushCodeLine("target_var = pd.DataFrame({'TargetVar': VF_TARGET})");
		rc=py.PushCodeLine("target_var[['id']] = 1");
		rc=py.PushCodeLine("target_var = target_var.reset_index().rename(columns={'index':'time'})");
		rc=py.PushCodeLine("df_target = target_var.dropna()");
		/* add user defined window size later */
		rc=py.PushCodeLine("window_size = &_windowsize");
		rc=py.PushCodeLine("step_size = 1");
		rc=py.PushCodeLine("df_shift, y = make_forecasting_frame(df_target['TargetVar'], kind='value', max_timeshift=window_size-1, rolling_direction=step_size)");
		rc=py.PushCodeLine("df_shift.drop(columns='kind',inplace=True)");
		rc=py.PushCodeLine("settings = EfficientFCParameters()");
		rc=py.PushCodeLine("extracted_features = extract_features(df_shift, column_id='id', column_sort='time', default_fc_parameters=settings)");
        rc=py.PushCodeLine("feature_dict={}");
        rc=py.PushCodeLine("for k,v in zip(extracted_features.columns.values, [f'_FEATURE{i}' for i in range(1,778)]):");
        rc=py.PushCodeLine("  feature_dict[k]=v");
		rc=py.PushCodeLine("impute(extracted_features)");
		rc=py.PushCodeLine("features_filtered = select_features(extracted_features, y)");
		rc=py.PushCodeLine("df_feature = features_filtered.droplevel(0).reset_index().rename(columns={'index':'time'})");
		rc=py.PushCodeLine("df_merged = target_var.merge(df_feature, on='time',how='left').drop(columns=['id','time'])");
		rc=py.PushCodeLine("df_merged_without_target = df_merged.drop(columns = ['TargetVar'])");
		rc=py.PushCodeLine("df_f = df_merged_without_target.rename(columns = feature_dict)");
		rc=py.PushCodeLine("for col in df_f.columns:");
		rc=py.PushCodeLine("  globals()[col] = df_f.loc[:,col].values");
		/* Run */
		rc=py.Run();
		runtime=py.GetRuntime();

		/* Store the execution and resource usage statistics logs */
		declare object pylog(OUTEXTLOG);
		rc=pylog.Collect(py, 'ALL');
		ENDSUBMIT;
	run;
	
	 proc cas;
    	table.droptable / name="GBMINPUTDATA" caslib="&vf_caslibOut" quiet=true;
    quit;
 
    data &vf_libOut.."GBMINPUTDATA"n (promote='yes') ;
       set &vf_libOut.."&vf_tableOutPrefix..fxInData"n ;
    run;

	
    /*Update the extracted feature macrovariable by eliminating columns with all missing values*/
    /* Step 1: Generate summary statistics */
    proc cas;
        simple.summary /
            table = {caslib = "&vf_caslibOut", name = "&vf_tableOutPrefix..fxInData"},
            casOut = {caslib = "&vf_caslibOut", name = "&vf_tableOutPrefix..summaryOut", replace = TRUE};
    quit;

    /* Step 2: Identify columns with all missing values using FedSQL */    
    proc cas;
	    sql_query = 'create table "' || "&vf_caslibOut" || '"."' || "&vf_tableOutPrefix" || '.allMissingCols" as ' ||
	                'select * ' ||
	                'from "' || "&vf_caslibOut" || '"."' || "&vf_tableOutPrefix" || '.summaryOut" ' ||
	                'where _NObs_ = 0;';
	    fedsql.execdirect / query=sql_query;
	quit;

    /* Step 3: Generate a macro variable storing all the names of missingg value columns*/
    proc sql noprint;
        select * into :allMissingCols separated by ' '
        from &vf_libOut.."&vf_tableOutPrefix..allMissingCols"n;
    quit;
    %put &allMissingCols;

    /* Step 4: Call macro fucntion to get &selectedfeatures with filtered column names*/
    %local i feature selectedfeatures;
    %let i = 1;
    %let feature = %scan(&EXFEATURES, &i);

    /*Check each feature against `&allMissingCols` and build `selectedfeatures` */
    %do %while(&feature ne);
        %if %index(&allMissingCols, &feature) = 0 %then %do;
            %if %length(&selectedfeatures) = 0 %then %do;
                %let selectedfeatures = &feature;
            %end;
            %else %do;
                %let selectedfeatures = &selectedfeatures &feature;
            %end;
        %end;
        %let i = %eval(&i + 1);
        %let feature = %scan(&EXFEATURES, &i);
    %end;

    /* Output the new macro variable */
    %put &selectedfeatures;

    /* Step 5: Update the input table, extend the features in forecasting horizon and keep only the selected features*/
	proc tsmodel data=&vf_libOut.."&vf_tableOutPrefix..fxInData"n
	    outarray=&vf_libOut.."&vf_tableOutPrefix..fxInData"n;
	    by &vf_byVars;
	    id &vf_timeID interval=&vf_timeIDInterval;
	    var &targetVar &vf_depVar &vf_indepVars %intervalFeatureVarList &selectedfeatures _roleVar;
	    require atsm;
	    submit;
	    declare object dataFrame(tsdf);
	    /* Dynamically add variables from selectedfeatures. */
	    %let i = 1;
	    %let featureName = %scan(&selectedfeatures, &i);
	    %do %while(&featureName ne );
	        if ^missing(&featureName[_LENGTH_-1-&vf_lead]) then do;
	        /* Initialize the dataFrame object once */
	        	rc = dataFrame.Initialize();
	            rc = dataFrame.AddX(&featureName);if rc < 0 then do; stop; end; 
				rc = dataFrame.setOption('HORIZON',&vf_horizonStart); if rc < 0 then do; stop; end; 
				rc = dataFrame.setOption('LEAD',&vf_lead); if rc < 0 then do; stop; end; 
	            rc = dataFrame.GetSeries("&featureName", &featureName, 'ADJUST', 'YES');if rc < 0 then do; stop; end; 
	        end;
	        %let i = %eval(&i + 1);
	        %let featureName = %scan(&selectedfeatures, &i);
	    %end;
	    endsubmit;
	run;


	/* Train the Light Gradient Boosting Model */
	proc lightgradboost 
			data=&vf_libOut.."&vf_tableOutPrefix..fxInData"n(where=(_roleVar=1)) 
			validdata=&vf_libOut.."&vf_tableOutPrefix..fxInData"n(where=(_roleVar=2)) 
			seed=12345;
		input &vf_byVars /level=NOMINAL;

		%if "&vf_indepVars" ne "" %then %do;
			input &vf_indepVars /level=INTERVAL;
		%end;

        %if "&selectedfeatures" ne "" %then %do;
			input &selectedfeatures /level=INTERVAL;
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
	            copyvars=(&vf_byVars &vf_timeID &targetVar &vf_depVar &vf_indepVars %intervalFeatureVarList &selectedfeatures);
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
	            var &targetVar &predictVar &vf_depVar &vf_indepVars %intervalFeatureVarList &selectedfeatures;
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
