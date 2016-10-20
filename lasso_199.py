import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.cross_validation import cross_val_score

#training data was preprocessed using weka to remove outliers. Around 50 outliers were found and were removed.

train = pd.read_csv("/home/tanushri/Documents/DWDM/train_or.csv")
test = pd.read_csv("/home/tanushri/Documents/DWDM/test.csv")
combo = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],test.loc[:,'MSSubClass':'SaleCondition']))

#Finding the list of numeric attributes and log transforming skewed numeric features

train["SalePrice"] = np.log1p(train["SalePrice"])
numeric_attributes = combo.dtypes[combo.dtypes != "object"].index
#finding features whose skewness is greater than 0.75
skwf = train[numeric_attributes].apply(lambda x: skew(x.dropna())) #compute skewness
skwf = skwf[skwf > 0.75]
skwf = skwf.index
combo[skwf] = np.log1p(combo[skwf])
##########################################################################################################
#filling NAs and converting categorical to numeric
#for some attributes NA represents absence of that property for example , for attributr 'Alley' 'NA' represents alley is not present in that particular house. Same applies to BsntExposure and some more features. these features should be handled in a different way as shown below.
x = combo.loc[np.logical_not(combo["LotFrontage"].isnull()), "LotArea"]
y = combo.loc[np.logical_not(combo["LotFrontage"].isnull()), "LotFrontage"]
t = (x <= 25000) & (y <= 150)
p = np.polyfit(x[t], y[t], 1)
combo.loc[combo['LotFrontage'].isnull(), 'LotFrontage'] = np.polyval(p, combo.loc[combo['LotFrontage'].isnull(), 'LotArea'])
combo.loc[combo.Alley.isnull(), 'Alley'] = 'NoAlley'
combo.loc[combo.MasVnrType.isnull(), 'MasVnrType'] = 'None' 
combo.loc[combo.MasVnrType == 'None', 'MasVnrArea'] = 0
combo.loc[combo.BsmtQual.isnull(), 'BsmtQual'] = 'NoBsmt'
combo.loc[combo.BsmtCond.isnull(), 'BsmtCond'] = 'NoBsmt'
combo.loc[combo.BsmtExposure.isnull(), 'BsmtExposure'] = 'NoBsmt'
combo.loc[combo.BsmtFinType1.isnull(), 'BsmtFinType1'] = 'NoBsmt'
combo.loc[combo.BsmtFinType2.isnull(), 'BsmtFinType2'] = 'NoBsmt'
combo.loc[combo.BsmtFinType1=='NoBsmt', 'BsmtFinSF1'] = 0
combo.loc[combo.BsmtFinType2=='NoBsmt', 'BsmtFinSF2'] = 0
combo.loc[combo.BsmtFinSF1.isnull(), 'BsmtFinSF1'] = combo.BsmtFinSF1.median()
combo.loc[combo.BsmtQual=='NoBsmt', 'BsmtUnfSF'] = 0
combo.loc[combo.BsmtUnfSF.isnull(), 'BsmtUnfSF'] = combo.BsmtUnfSF.median()
combo.loc[combo.BsmtQual=='NoBsmt', 'TotalBsmtSF'] = 0
combo.loc[combo.FireplaceQu.isnull(), 'FireplaceQu'] = 'NoFireplace'
combo.loc[combo.GarageType.isnull(), 'GarageType'] = 'NoGarage'
combo.loc[combo.GarageFinish.isnull(), 'GarageFinish'] = 'NoGarage'
combo.loc[combo.GarageQual.isnull(), 'GarageQual'] = 'NoGarage'
combo.loc[combo.GarageCond.isnull(), 'GarageCond'] = 'NoGarage'
combo.loc[combo.BsmtFullBath.isnull(), 'BsmtFullBath'] = 0
combo.loc[combo.BsmtHalfBath.isnull(), 'BsmtHalfBath'] = 0
combo.loc[combo.KitchenQual.isnull(), 'KitchenQual'] = 'TA'
combo.loc[combo.MSZoning.isnull(), 'MSZoning'] = 'RL'
combo.loc[combo.Utilities.isnull(), 'Utilities'] = 'AllPub'
combo.loc[combo.Exterior1st.isnull(), 'Exterior1st'] = 'VinylSd'
combo.loc[combo.Exterior2nd.isnull(), 'Exterior2nd'] = 'VinylSd'
combo.loc[combo.Functional.isnull(), 'Functional'] = 'Typ'
combo.loc[combo.SaleCondition.isnull(), 'SaleCondition'] = 'Normal'
combo.loc[combo.SaleCondition.isnull(), 'SaleType'] = 'WD'
combo.loc[combo['PoolQC'].isnull(), 'PoolQC'] = 'NoPool'
combo.loc[combo['Fence'].isnull(), 'Fence'] = 'NoFence'
combo.loc[combo['MiscFeature'].isnull(), 'MiscFeature'] = 'None'
combo.loc[combo['Electrical'].isnull(), 'Electrical'] = 'SBrkr'
combo.loc[combo['GarageArea'].isnull(), 'GarageArea'] = combo.loc[combo['GarageType']=='Detchd', 'GarageArea'].mean()
combo.loc[combo['GarageCars'].isnull(), 'GarageCars'] = combo.loc[combo['GarageType']=='Detchd', 'GarageCars'].median()



###########################################################################################################
#Convert categorical variable into dummy/indicator variables
combo = pd.get_dummies(combo)
#Fill NA/NaN values using the specified method
combo = combo.fillna(combo.mean())
#creating matrices for sklearn:
X_train = combo[:train.shape[0]]
X_test = combo[train.shape[0]:]
y = train.SalePrice

###########################################################################################################

def rmse_cv(model):
    rmse= np.sqrt(cross_val_score(model, X_train, y, scoring="mean_squared_error", cv = 5))
    return(rmse)



alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]

lassom = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y)
rmse_cv(lassom).mean()
preds = pd.DataFrame({"preds":lassom.predict(X_train), "true":y})
preds["residuals"] = preds["true"] - preds["preds"]
lasso_preds = np.expm1(lassom.predict(X_test))
preds = lasso_preds
solution = pd.DataFrame({"id":test.Id, "SalePrice":preds})
solution.to_csv("ridge_sol_final.csv", index = False)

##################################################################################################################################
# the lasso predictions are combined with xgboost predictions. Lasso predictions are given 0.8 weight , xgboost predictions are given 0.2 weight 
#Finally the merged predictions were submitted.
