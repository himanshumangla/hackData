import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from pprint import pprint
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
import itertools
from sklearn.cross_validation import train_test_split
import statsmodels.api as sm


#This is exponential
def select_features(df0,**kwargs):

    best_score = []
    best_std   = []
    best_comb  = []
    nfeatures  = 18
    iterable   = range(nfeatures)
    for intercept in [True,False]:
        model  = LogisticRegression(fit_intercept = intercept, penalty= penalty, dual=False, C = cval)
        for s in xrange(len(iterable)+1):
            for comb in itertools.combinations(iterable, s):
                print('%d \n' %s)
                if len(comb) > 0:
                    Xsel = []
                    for patient in Xall:
                        Xsel.append([patient[ind] for ind in comb])
                    this_scores = cross_val_score(model, Xsel, y=yall, cv=3)
                    score_mean  = np.mean(this_scores)
                    score_std   = np.std(this_scores)
                    comb1       = list(comb)
                    if intercept: comb1.append(nfeatures)
                    if len(best_score) > 0: 
                        if score_mean > best_score[0]:
                            best_score = []
                            best_std   = []
                            best_comb  = []
                            best_score.append(score_mean)
                            best_std.append(score_std)
                            best_comb.append(comb1)
                        elif score_mean == best_score[0]:
                            print 'equal scoress'
                            best_score.append(score_mean)
                            best_std.append(score_std)
                            best_comb.append(comb1)
                    else:
                        best_score.append(score_mean)
                        best_std.append(score_std)
                        best_comb.append(comb1)

    new_columns_3 = new_columns_2[1:]
    new_columns_3.append("intercept")
    num_ties = len(best_score)
    print('\n\n num ties = %d' %num_ties)
    for ind in range(num_ties):
        comb1 = best_comb[ind][:]
        if nfeatures in comb1:
            intercept = True
            comb1.remove(nfeatures)
        else:
            intercept = False
        model = LogisticRegression(fit_intercept = intercept, penalty = penalty, dual = False, C = cval)
        Xsel  = []
        for patient in Xall:
            Xsel.append([patient[i] for i in comb1])
        lrfit = model.fit(Xsel,yall)
        print('\nResults for combination %s:' %best_comb[ind])
        print('LogisticRegression score on full data set:      %f' % lrfit.score(Xsel,yall))
        print('LogisticRegression score from cross-validation: %f +/- %f' % (best_score[ind],best_std[ind]))
        print('LogisticRegression coefficients:')
        coeff = model.coef_.tolist()[0]
        if intercept: coeff.append(model.intercept_)
        for jnd in range(len(coeff)):
            print '%s : %8.5f' % (new_columns_3[best_comb[ind][jnd]].rjust(9),coeff[jnd])


def select_features2(df0,**kwargs):
    """Selects features for a logistic regression model. Starts by including all features, and then
    eliminates non-significant features one-by-one, in such a way as to minimize the increase in 
    deviance after each elimination."""
    if "verbose" in kwargs: 
        vb = kwargs["verbose"]
    else:
        vb = 0
    if "method" in kwargs:
        md = kwargs["method"]
    else:
        md = "newton"
    feature_names   = list(df0.columns)      # Get data frame column names, these are the features to be fitted.
    feature_names.append("intercept")        # Add an intercept feature.
    nfeatures       = len(feature_names) - 1 # First feature is response variable, so doesn't count.
    saved_columns   = range(1,nfeatures+1)
    dropped_columns = []
    while len(saved_columns)>1:
        df1              = df0.copy()
        df1["intercept"] = 1.0
        df1              = df1.drop(df1.columns[dropped_columns],axis=1)
        features         = df1[df1.columns[1:]]
        response         = df1[df1.columns[0]]
        model            = sm.Logit(response, features, missing="drop")
        result           = model.fit(method=md, maxiter=100, disp=0)
        if len(saved_columns)==nfeatures:
            if vb >= 1:
                print '\nInitial -log(L)=%8.5f; fit method = %s\n' %(-result.llf,md)
        if all(np.abs(tval) >= 2.0 for tval in result.tvalues):
            if vb >= 1:
                print '\nFinal selection includes %i features:\n' % len(saved_columns)
                print result.summary()
            break
        else:
            llfmin = 999999.0
            colmin = -100
            for col in saved_columns:
                dropped_columns.append(col)
                df1              = df0.copy()
                df1["intercept"] = 1.0
                df1              = df1.drop(df1.columns[dropped_columns],axis=1)
                features         = df1[df1.columns[1:]]
                response         = df1[df1.columns[0]]
                model            = sm.Logit(response, features, missing="drop")
                result           = model.fit(method=md, maxiter=100, disp=0)
                if -result.llf < llfmin:
                    llfmin = -result.llf
                    colmin = col
                dropped_columns.remove(col)
            if vb >= 2:
                print 'Dropping %s, -log(L)=%8.5f...' % (feature_names[colmin].rjust(9),llfmin)
            saved_columns.remove(colmin)
            dropped_columns.append(colmin)
    else:
        if vb >= 1:
            print 'Final selection has no features left'
    return result,saved_columns



def train_and_test(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    columns = ["age", "sex", "cp", "restbp", "chol", "fbs", "restecg", 
               "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
    df0     = pd.read_table("heartrisk/static/heart_disease_all14.csv", sep=',', header=None, names=columns)


    #Convert categorical variables into dummy variables.
    df      = df0.copy()
    dummies = pd.get_dummies(df["cp"],prefix="cp")
    df      = df.join(dummies)
    del df["cp"]
    del df["cp_4.0"]
    df      = df.rename(columns = {"cp_1.0":"cp_1","cp_2.0":"cp_2","cp_3.0":"cp_3"})

    dummies = pd.get_dummies(df["restecg"],prefix="recg")
    df      = df.join(dummies)
    del df["restecg"]
    del df["recg_0.0"]
    df      = df.rename(columns = {"recg_1.0":"recg_1","recg_2.0":"recg_2"})

    dummies = pd.get_dummies(df["slope"],prefix="slope")
    df      = df.join(dummies)
    del df["slope"]
    del df["slope_2.0"]
    df      = df.rename(columns = {"slope_1.0":"slope_1","slope_3.0":"slope_3"})

    dummies = pd.get_dummies(df["thal"],prefix="thal")
    df      = df.join(dummies)
    del df["thal"]
    del df["thal_3.0"]
    df      = df.rename(columns = {"thal_6.0":"thal_6","thal_7.0":"thal_7"})

    # Replace target variable num
    df["num"].replace(to_replace=[1,2,3,4],value=1,inplace=True)
    df      = df.rename(columns = {"num":"hd"})

    # New list of column labels
    new_columns_1 = ["age", "sex", "restbp", "chol", "fbs", "thalach", 
                     "exang", "oldpeak", "ca", "hd", "cp_1", "cp_2",
                     "cp_3", "recg_1", "recg_2", "slope_1", "slope_3",
                     "thal_6", "thal_7"]

    print '\nNumber of patients in dataframe: %i, with disease: %i, without disease: %i\n'       % (len(df.index),len(df[df.hd==1].index),len(df[df.hd==0].index))


    #Standardize the data
    stdcols = ["age","restbp","chol","thalach","oldpeak"]
    nrmcols = ["ca"]
    stddf   = df.copy()
    stddf[stdcols] = stddf[stdcols].apply(lambda x: (x-x.mean())/x.std())
    stddf[nrmcols] = stddf[nrmcols].apply(lambda x: (x-x.mean())/(x.max()-x.min()))

    new_columns_2 = new_columns_1[:9] + new_columns_1[10:]
    new_columns_2.insert(0,new_columns_1[9])
    stddf = stddf.reindex(columns=new_columns_2)


    #Convert dataframe into lists for use by classifiers
    yall = stddf["hd"]
    Xall = stddf[new_columns_2[1:]].values



    penalty = "l1"
    cval    = 1000.0
    alpha   = 0.0

    train, test = train_test_split(stddf, test_size=0.25, random_state=1)
    model, selected_combo = select_features2(train, verbose=0, method="bfgs")

    feature_names = list(stddf.columns)
    print('Partitioned features: %s' %[feature_names[i] for i in selected_combo])

    test_true = list(test.hd)

    #Removing same features from test data
    nfeatures = len(stddf.columns)
    drop = [i for i in range(nfeatures) if i not in selected_combo]
    test = test.drop(test.columns[drop], axis=1)

    #Predicting
    test_pred = model.predict(test, linear=False)
    true_pred = zip(test_true, test_pred)

    #Calculating scores
    for ind in [true_pred]:
        yy = sum([1 for [tr,pr] in ind if tr==1 and pr>=0.5])
        yn = sum([1 for [tr,pr] in ind if tr==1 and pr<0.5])
        nn = sum([1 for [tr,pr] in ind if tr==0 and pr<0.5])
        ny = sum([1 for [tr,pr] in ind if tr==0 and pr>=0.5])

    acc = float(yy+nn)/(yy+yn+ny+nn)   # fraction of predictions that are correct
    pr1 = float(yy)/(yy+ny)            # fraction of disease predictions that are correct
    pr2 = float(nn)/(nn+yn)            # fraction of no-disease predictions that are correct
    rl1 = float(yy)/(yy+yn)            # fraction of disease cases that are identified
    rl2 = float(nn)/(nn+ny)            # fraction of no-disease cases that are identified


    print('Accuracy:                [%5.3f]' %acc)
    print('Precision on disease:    [%5.3f]' %pr1)
    print('Precision on no-disease: [%5.3f]' %pr2)
    print('Recall on disease:       [%5.3f]' %rl1)
    print('Recall on no-disease:    [%5.3f]' %rl2)

    #Testing
    test_df  = []
    test_df.append(age)
    test_df.append(sex)
    test_df.append(cp)
    test_df.append(trestbps)
    test_df.append(chol)
    test_df.append(fbs)
    test_df.append(restecg)
    test_df.append(thalach)
    test_df.append(exang)
    test_df.append(oldpeak)
    test_df.append(slope)
    test_df.append(ca)
    test_df.append(thal)
    test_df.append(u'0')
    test_df = [float(i) for i in test_df]   
    print test_df

    df0.loc[-1] = test_df  # adding a row
    df0.index = df0.index + 1  # shifting index
    result = df0.sort()  # sorting by index
        
    dfx      = result.copy()
    dummies = pd.get_dummies(dfx["cp"],prefix="cp")
    dfx      = dfx.join(dummies)
    del dfx["cp"]
    del dfx["cp_4.0"]
    dfx      = dfx.rename(columns = {"cp_1.0":"cp_1","cp_2.0":"cp_2","cp_3.0":"cp_3"})

    dummies = pd.get_dummies(dfx["restecg"],prefix="recg")
    dfx      = dfx.join(dummies)
    del dfx["restecg"]
    del dfx["recg_0.0"]
    dfx      = dfx.rename(columns = {"recg_1.0":"recg_1","recg_2.0":"recg_2"})

    dummies = pd.get_dummies(dfx["slope"],prefix="slope")
    dfx      = dfx.join(dummies)
    del dfx["slope"]
    del dfx["slope_2.0"]
    dfx      = dfx.rename(columns = {"slope_1.0":"slope_1","slope_3.0":"slope_3"})

    dummies = pd.get_dummies(dfx["thal"],prefix="thal")
    dfx      = dfx.join(dummies)
    del dfx["thal"]
    del dfx["thal_3.0"]
    dfx      = dfx.rename(columns = {"thal_6.0":"thal_6","thal_7.0":"thal_7"})

    # Replace response variable values and rename
    dfx["num"].replace(to_replace=[1,2,3,4],value=1,inplace=True)
    dfx      = dfx.rename(columns = {"num":"hd"})

    # New list of column labels after the above operations
    new_columns_1 = ["age", "sex", "restbp", "chol", "fbs", "thalach", 
                     "exang", "oldpeak", "ca", "hd", "cp_1", "cp_2",
                     "cp_3", "recg_1", "recg_2", "slope_1", "slope_3",
                     "thal_6", "thal_7"]


    # Standardize the dataframe
    stdcols = ["age","restbp","chol","thalach","oldpeak"]
    nrmcols = ["ca"]
    stddfx   = dfx.copy()
    stddfx[stdcols] = stddfx[stdcols].apply(lambda x: (x-x.mean())/x.std())
    stddfx[nrmcols] = stddfx[nrmcols].apply(lambda x: (x-x.mean())/(x.max()-x.min()))

    new_columns_2 = new_columns_1[:9] + new_columns_1[10:]
    new_columns_2.insert(0,new_columns_1[9])
    stddfx = stddfx.reindex(columns=new_columns_2)

    stddfx = stddfx.drop(stddfx.columns[drop], axis=1)

    #stddfx.fillna(0, inplace=True)
    hdx_pred  = model.predict(stddfx,linear=False)
    return hdx_pred[0]



