import yahoo_fin.stock_info as si

from datetime import date
import numpy as np

# from https://medium.com/@maneesh1.kumar/correlating-vix-and-spy-a-statistical-study-using-pythons-linear-regression-9261bfd9dd71

def load_SPY_VIX(start_date = date(2000,1,1),
                end_date = date(2023,1,1)
                ):
    dfSPY = si.get_data('SPY',start_date=start_date, end_date=end_date).round(1)[['adjclose']]
    dfVIX = si.get_data('^VIX',start_date=start_date, end_date=end_date).round(1)[['adjclose']]
    dfSPY.columns=['SPY']
    dfVIX.columns=['VIX']
    df=dfSPY.join(dfVIX).dropna()
    return df

def sim_VS(params, dt, N, S_init=100, V_init=-0.5, dw=None):
    # assuming constant mu
    mu, lambda_v, theta, sigma_v, ro = params
    ro_hat = np.sqrt(1-ro**2)
    
    # independently sampled dw are stacked together
    if dw is None:
        dw = [np.sqrt(dt) * np.random.normal(size=N)
              for i in range(2)]
    
    V = np.zeros(N)
    S = np.zeros(N)
    V[0] = V_init
    S[0] = S_init
    
    for i in range(1, N):
        V[i] = V[i-1] + lambda_v*(theta - V[i-1])*dt + sigma_v*ro*dw[0][i-1] + sigma_v*ro_hat*dw[1][i-1]
        S[i] = S[i-1] + S[i-1]*(np.exp(V[i-1])*mu*dt + np.exp(V[i-1])*dw[0][i-1])
    
    return S, V, dw

def sim_VSmu(params, dt, N, S_init=100, V_init=-0.5, mu_init=10, dw=None):
    rho, sigma_v, sigma_mu, lambda_v, lambda_mu, theta, theta_mu = params
    rho_hat = np.sqrt(1-rho**2)
    
    # independently sampled dw are stacked together
    if dw is None:
        dw = [np.sqrt(dt) * np.random.normal(size=N)
              for i in range(3)]
    
    S, V, mu = np.zeros((3, N))
    V[0] = V_init
    S[0] = S_init
    mu[0] = mu_init
    
    for i in range(1, N):
        mu[i] = mu[i-1] + lambda_mu*(theta_mu - mu[i-1])*dt + sigma_mu*dw[2][i-1]
        V[i] = V[i-1] + lambda_v*(theta - V[i-1])*dt + sigma_v*rho*dw[0][i-1] + sigma_v*rho_hat*dw[1][i-1]
        S[i] = S[i-1] + S[i-1]*(np.exp(V[i-1])*mu[i-1]*dt + np.exp(V[i-1])*dw[0][i-1])
    
    return S, V, mu, dw
