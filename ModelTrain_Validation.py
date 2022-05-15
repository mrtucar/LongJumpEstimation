
Model = "Ann"
Fold = []
Noron_1 = []
Noron_2 = []
Gercek = []
Tahmin = []
MSE = []
RMSE = []
MAE = []
EVS = []
MSLE = []
MABSOLERR = []
R2 = []
TIME = []
for fold in range(1,6):
    
    dataTrain  = pd.read_csv(f"Train_{fold}.csv")
    dataTest  = pd.read_csv(f"Test_{fold}.csv")
    
    X_train  = dataTrain[["V1","V2","Vloss","Age"]].values
    y_train  = dataTrain["Distance"].values
    
    X_test  = dataTest[["V1","V2","Vloss","Age"]].values
    y_test  = dataTest["Distance"].values    
    
    for noron_1 in range(5,55,5):
        for noron_2 in range (5,55,5):
            
            
            model =  MLPRegressor( hidden_layer_sizes=(noron_1,noron_2),  activation='relu', solver='adam', alpha=0.001,
                                  batch_size='auto',learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, 
                                  shuffle=True, random_state=21, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, 
                                  nesterovs_momentum=True,early_stopping=False, validation_fraction=0.1, beta_1=0.9, 
                                  beta_2=0.999, epsilon=1e-08)
            start_time = time.time()
            model.fit(X_train,y_train)
            TIME.append((time.time() - start_time))
            y_head = model.predict(X_test)
            mse = mean_squared_error(y_test, y_head)
            rmse = sqrt(mse)
            mae = mean_absolute_error(y_test, y_head)
            evs = explained_variance_score(y_test, y_head)
            msle = mean_squared_log_error(y_test, y_head)
            mabsolerr = median_absolute_error(y_test, y_head)
            r2 = r2_score(y_test, y_head)
            
            Fold.append(fold)
            Noron_1.append(noron_1)
            Noron_2.append(noron_2)
            Gercek.append(y_test)
            Tahmin.append(y_head)
            MSE.append(mse)
            RMSE.append(rmse)
            MAE.append(mae)
            EVS.append(evs)
            MSLE.append(msle)
            MABSOLERR.append(mabsolerr)
            R2.append(r2)
            print(f"{fold}-{noron_1}-{noron_2}")
            
dfAnnSonuclar = pd.DataFrame({
    "Fold " : Fold,
    "Noron_1":Noron_1,
    "Noron_2":Noron_2,
    "Gercek":Gercek,
    "Tahmin":Tahmin,
    "MSE":MSE,
    "RMSE":RMSE,
    "MAE":MAE,
    "EVS":EVS,
    "MSLE":MSLE,
    "MABSOLERR":MABSOLERR,
    "R2":R2,
    "TIME":TIME
    })

dfAnnSonuclar.to_excel("Ann_Sonuclar.xlsx")
#%%            
Model = "Ridge Regressor"
Fold = []
ALPHA = []
Gercek = []
Tahmin = []
MSE = []
RMSE = []
MAE = []
EVS = []
MSLE = []
MABSOLERR = []
R2 = []
TIME = []

for fold in range(1,6):
    
    dataTrain  = pd.read_csv(f"Train_{fold}.csv")
    dataTest  = pd.read_csv(f"Test_{fold}.csv")
    
    X_train  = dataTrain[["V1","V2","Vloss","Age"]].values
    y_train  = dataTrain["Distance"].values
    
    X_test  = dataTest[["V1","V2","Vloss","Age"]].values
    y_test  = dataTest["Distance"].values    
    
    for i in range(1,31,1):
            model = Ridge(alpha=(i/10) )
            start_time = time.time()
            model.fit(X_train,y_train)
            TIME.append((time.time() - start_time))
            y_head = model.predict(X_test)
            mse = mean_squared_error(y_test, y_head)
            rmse = sqrt(mse)
            mae = mean_absolute_error(y_test, y_head)
            evs = explained_variance_score(y_test, y_head)
            msle = mean_squared_log_error(y_test, y_head)
            mabsolerr = median_absolute_error(y_test, y_head)
            r2 = r2_score(y_test, y_head)
            
            Fold.append(fold)
            ALPHA.append(i/10)
            Gercek.append(y_test)
            Tahmin.append(y_head)
            MSE.append(mse)
            RMSE.append(rmse)
            MAE.append(mae)
            EVS.append(evs)
            MSLE.append(msle)
            MABSOLERR.append(mabsolerr)
            R2.append(r2)
            
            print(model)
            
dfAnnSonuclar = pd.DataFrame({
    "Fold " : Fold,
    "ALPHA":ALPHA,
    "Gercek":Gercek,
    "Tahmin":Tahmin,
    "MSE":MSE,
    "RMSE":RMSE,
    "MAE":MAE,
    "EVS":EVS,
    "MSLE":MSLE,
    "MABSOLERR":MABSOLERR,
    "R2":R2,
    "TIME":TIME    
    })

dfAnnSonuclar.to_excel("RR_Sonuclar.xlsx")            
            
#%%
Model = "KNN"
Fold = []
NEIGHBORS = []
Gercek = []
Tahmin = []
MSE = []
RMSE = []
MAE = []
EVS = []
MSLE = []
MABSOLERR = []
R2 = []
TIME = []

for fold in range(1,6):
    
    dataTrain  = pd.read_csv(f"Train_{fold}.csv")
    dataTest  = pd.read_csv(f"Test_{fold}.csv")
    
    X_train  = dataTrain[["V1","V2","Vloss","Age"]].values
    y_train  = dataTrain["Distance"].values
    
    X_test  = dataTest[["V1","V2","Vloss","Age"]].values
    y_test  = dataTest["Distance"].values    
    
    for i in range(1,21,1):
            model = KNeighborsRegressor(n_neighbors=i,weights='uniform')
            start_time = time.time()
            model.fit(X_train,y_train)
            TIME.append((time.time() - start_time))
            y_head = model.predict(X_test)
            mse = mean_squared_error(y_test, y_head)
            rmse = sqrt(mse)
            mae = mean_absolute_error(y_test, y_head)
            evs = explained_variance_score(y_test, y_head)
            msle = mean_squared_log_error(y_test, y_head)
            mabsolerr = median_absolute_error(y_test, y_head)
            r2 = r2_score(y_test, y_head)
            
            Fold.append(fold)
            NEIGHBORS.append(i)
            Gercek.append(y_test)
            Tahmin.append(y_head)
            MSE.append(mse)
            RMSE.append(rmse)
            MAE.append(mae)
            EVS.append(evs)
            MSLE.append(msle)
            MABSOLERR.append(mabsolerr)
            R2.append(r2)
            
            print(model)
            
dfAnnSonuclar = pd.DataFrame({
    "Fold " : Fold,
    "NEIGHBORS":NEIGHBORS,
    "Gercek":Gercek,
    "Tahmin":Tahmin,
    "MSE":MSE,
    "RMSE":RMSE,
    "MAE":MAE,
    "EVS":EVS,
    "MSLE":MSLE,
    "MABSOLERR":MABSOLERR,
    "R2":R2,
    "TIME":TIME    
    })

dfAnnSonuclar.to_excel("KNN_Sonuclar.xlsx")
#%%
Model = "DecisionTree"
Fold = []
NEIGHBORS = []
Gercek = []
Tahmin = []
MSE = []
RMSE = []
MAE = []
EVS = []
MSLE = []
MABSOLERR = []
R2 = []
TIME = []

for fold in range(1,6):
    
    dataTrain  = pd.read_csv(f"Train_{fold}.csv")
    dataTest  = pd.read_csv(f"Test_{fold}.csv")
    
    X_train  = dataTrain[["V1","V2","Vloss","Age"]].values
    y_train  = dataTrain["Distance"].values
    
    X_test  = dataTest[["V1","V2","Vloss","Age"]].values
    y_test  = dataTest["Distance"].values    
    
    model = DecisionTreeRegressor()
    start_time = time.time()
    model.fit(X_train,y_train)
    TIME.append((time.time() - start_time))
    y_head = model.predict(X_test)
    mse = mean_squared_error(y_test, y_head)
    rmse = sqrt(mse)
    mae = mean_absolute_error(y_test, y_head)
    evs = explained_variance_score(y_test, y_head)
    msle = mean_squared_log_error(y_test, y_head)
    mabsolerr = median_absolute_error(y_test, y_head)
    r2 = r2_score(y_test, y_head)
     
    Fold.append(fold)
    #NEIGHBORS.append(i)
    Gercek.append(y_test)
    Tahmin.append(y_head)
    MSE.append(mse)
    RMSE.append(rmse)
    MAE.append(mae)
    EVS.append(evs)
    MSLE.append(msle)
    MABSOLERR.append(mabsolerr)
    R2.append(r2)
     
    print(model)
            
dfAnnSonuclar = pd.DataFrame({
    "Fold " : Fold,
    #"NEIGHBORS":NEIGHBORS,
    "Gercek":Gercek,
    "Tahmin":Tahmin,
    "MSE":MSE,
    "RMSE":RMSE,
    "MAE":MAE,
    "EVS":EVS,
    "MSLE":MSLE,
    "MABSOLERR":MABSOLERR,
    "R2":R2,
    "TIME":TIME    
    })

dfAnnSonuclar.to_excel("DT_Sonuclar.xlsx")        
#%%    
Model = "GB"
Fold = []
ESTIMATOR = []
Gercek = []
Tahmin = []
MSE = []
RMSE = []
MAE = []
EVS = []
MSLE = []
MABSOLERR = []
R2 = []
TIME = []

for fold in range(1,6):
    
    dataTrain  = pd.read_csv(f"Train_{fold}.csv")
    dataTest  = pd.read_csv(f"Test_{fold}.csv")
    
    X_train  = dataTrain[["V1","V2","Vloss","Age"]].values
    y_train  = dataTrain["Distance"].values
    
    X_test  = dataTest[["V1","V2","Vloss","Age"]].values
    y_test  = dataTest["Distance"].values    
    
    for i in [50,100,200,300,400,500,600,700,800,900,1000]:
                
            model = GradientBoostingRegressor(n_estimators=i)
            
            start_time = time.time()
            model.fit(X_train,y_train)
            TIME.append((time.time() - start_time))
            y_head = model.predict(X_test)
            mse = mean_squared_error(y_test, y_head)
            rmse = sqrt(mse)
            mae = mean_absolute_error(y_test, y_head)
            evs = explained_variance_score(y_test, y_head)
            msle = mean_squared_log_error(y_test, y_head)
            mabsolerr = median_absolute_error(y_test, y_head)
            r2 = r2_score(y_test, y_head)
            
            Fold.append(fold)
            ESTIMATOR.append(i)
            Gercek.append(y_test)
            Tahmin.append(y_head)
            MSE.append(mse)
            RMSE.append(rmse)
            MAE.append(mae)
            EVS.append(evs)
            MSLE.append(msle)
            MABSOLERR.append(mabsolerr)
            R2.append(r2)
            
            print(model)
            
dfAnnSonuclar = pd.DataFrame({
    "Fold " : Fold,
    "ESTIMATOR":ESTIMATOR,
    "Gercek":Gercek,
    "Tahmin":Tahmin,
    "MSE":MSE,
    "RMSE":RMSE,
    "MAE":MAE,
    "EVS":EVS,
    "MSLE":MSLE,
    "MABSOLERR":MABSOLERR,
    "R2":R2,
    "TIME":TIME    
    })

dfAnnSonuclar.to_excel("GB_Sonuclar.xlsx")
#%%
Model = "RF"
Fold = []
ESTIMATOR = []
Gercek = []
Tahmin = []
MSE = []
RMSE = []
MAE = []
EVS = []
MSLE = []
MABSOLERR = []
R2 = []
TIME = []
for fold in range(1,6):
    
    dataTrain  = pd.read_csv(f"Train_{fold}.csv")
    dataTest  = pd.read_csv(f"Test_{fold}.csv")
    
    X_train  = dataTrain[["V1","V2","Vloss","Age"]].values
    y_train  = dataTrain["Distance"].values
    
    X_test  = dataTest[["V1","V2","Vloss","Age"]].values
    y_test  = dataTest["Distance"].values    
    
    for i in [50,100,200,300,400,500,600,700,800,900,1000]:
            model = RandomForestRegressor(n_estimators=i)
            start_time = time.time()
            model.fit(X_train,y_train)
            TIME.append((time.time() - start_time))
            y_head = model.predict(X_test)
            mse = mean_squared_error(y_test, y_head)
            rmse = sqrt(mse)
            mae = mean_absolute_error(y_test, y_head)
            evs = explained_variance_score(y_test, y_head)
            msle = mean_squared_log_error(y_test, y_head)
            mabsolerr = median_absolute_error(y_test, y_head)
            r2 = r2_score(y_test, y_head)
            
            Fold.append(fold)
            ESTIMATOR.append(i)
            Gercek.append(y_test)
            Tahmin.append(y_head)
            MSE.append(mse)
            RMSE.append(rmse)
            MAE.append(mae)
            EVS.append(evs)
            MSLE.append(msle)
            MABSOLERR.append(mabsolerr)
            R2.append(r2)
            
            print(model)
            
dfAnnSonuclar = pd.DataFrame({
    "Fold " : Fold,
    "ESTIMATOR":ESTIMATOR,
    "Gercek":Gercek,
    "Tahmin":Tahmin,
    "MSE":MSE,
    "RMSE":RMSE,
    "MAE":MAE,
    "EVS":EVS,
    "MSLE":MSLE,
    "MABSOLERR":MABSOLERR,
    "R2":R2,
    "TIME":TIME    
    })

dfAnnSonuclar.to_excel("RF_Sonuclar.xlsx")

#%%
Model = "LinearRegression"
Fold = []
NEIGHBORS = []
Gercek = []
Tahmin = []
MSE = []
RMSE = []
MAE = []
EVS = []
MSLE = []
MABSOLERR = []
R2 = []
TIME = []


for fold in range(1,6):
    
    dataTrain  = pd.read_csv(f"Train_{fold}.csv")
    dataTest  = pd.read_csv(f"Test_{fold}.csv")
    
    X_train  = dataTrain[["V1","V2","Vloss","Age"]].values
    y_train  = dataTrain["Distance"].values
    
    X_test  = dataTest[["V1","V2","Vloss","Age"]].values
    y_test  = dataTest["Distance"].values    
    
    model = LinearRegression()
    
    start_time = time.time()
    model.fit(X_train,y_train)
    TIME.append((time.time() - start_time))

    y_head = model.predict(X_test)
    mse = mean_squared_error(y_test, y_head)
    rmse = sqrt(mse)
    mae = mean_absolute_error(y_test, y_head)
    evs = explained_variance_score(y_test, y_head)
    msle = mean_squared_log_error(y_test, y_head)
    mabsolerr = median_absolute_error(y_test, y_head)
    r2 = r2_score(y_test, y_head)
     
    Fold.append(fold)
    #NEIGHBORS.append(i)
    Gercek.append(y_test)
    Tahmin.append(y_head)
    MSE.append(mse)
    RMSE.append(rmse)
    MAE.append(mae)
    EVS.append(evs)
    MSLE.append(msle)
    MABSOLERR.append(mabsolerr)
    R2.append(r2)
     
    print(model)
            
dfAnnSonuclar = pd.DataFrame({
    "Fold " : Fold,
    #"NEIGHBORS":NEIGHBORS,
    "Gercek":Gercek,
    "Tahmin":Tahmin,
    "MSE":MSE,
    "RMSE":RMSE,
    "MAE":MAE,
    "EVS":EVS,
    "MSLE":MSLE,
    "MABSOLERR":MABSOLERR,
    "R2":R2,
    "TIME":TIME    
    })

dfAnnSonuclar.to_excel("LR_Sonuclar.xlsx")
#%% Isik Hocalar Parametre ile Hesaplama
Model = "IsikHocalarParametre"
Fold = []
Gercek = []
Tahmin = []
MSE = []
RMSE = []
MAE = []
EVS = []
MSLE = []
MABSOLERR = []
R2 = []
TIME = []


for fold in range(1,6):
    
    dataTrain  = pd.read_csv(f"Train_{fold}.csv")
    dataTest  = pd.read_csv(f"Test_{fold}.csv")
    
    X_train  = dataTrain[["V1","V2","Vloss","Age"]].values
    y_train  = dataTrain["Distance"].values
    
    X_test  = dataTest[["V1","V2","Vloss","Age"]].values
    y_test  = dataTest["Distance"].values    
    
    X_test = scaler.inverse_transform(X_test)

    y_head =  ( ( (X_test[:,0] + X_test[:,1]) / 2 )  *    1.067 ) - 3.184 
    mse = mean_squared_error(y_test, y_head)
    rmse = sqrt(mse)
    mae = mean_absolute_error(y_test, y_head)
    evs = explained_variance_score(y_test, y_head)
    msle = mean_squared_log_error(y_test, y_head)
    mabsolerr = median_absolute_error(y_test, y_head)
    r2 = r2_score(y_test, y_head)
     
    Fold.append(fold)
    #NEIGHBORS.append(i)
    Gercek.append(y_test)
    Tahmin.append(y_head)
    MSE.append(mse)
    RMSE.append(rmse)
    MAE.append(mae)
    EVS.append(evs)
    MSLE.append(msle)
    MABSOLERR.append(mabsolerr)
    R2.append(r2)
     
    print(model)
            
dfAnnSonuclar = pd.DataFrame({
    "Fold " : Fold,
    "Gercek":Gercek,
    "Tahmin":Tahmin,
    "MSE":MSE,
    "RMSE":RMSE,
    "MAE":MAE,
    "EVS":EVS,
    "MSLE":MSLE,
    "MABSOLERR":MABSOLERR,
    "R2":R2,
    "TIME":TIME    
    })

dfAnnSonuclar.to_excel("LR_Parametre.xlsx")
   
    
    
    