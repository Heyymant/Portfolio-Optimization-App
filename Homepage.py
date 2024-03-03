#Import the python libraries
import streamlit as st 
import pandas as pd
import numpy as np
from pandas_datareader import data as web
import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf
# import matplotlib.pyplot as plt
import plotly.graph_objects as go
# import plotly.figure_factory as ff

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import plotting
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
# import plotly.express as plt
import copy
import plotly.express as px
import seaborn as sns
from datetime import datetime
from io import BytesIO
# plt.style.use('fivethirtyeight')

@st.cache_data
def bg_png(png):
    """
    function to display png as bg
    ----------
    png_file: png -> the background image in local folder
    """
    st.markdown(f"""
         <style>
         .stApp {{
             background: url("https://images.unsplash.com/photo-1640340434855-6084b1f4901c?q=80&w=2000&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
             background-size: cover
             opacity :
         }}
         
        [data-testid = "stHeader"]{{
            background : rgba(0,0,0,0);
        }}
        [data-testid="stSidebar"] > div:first-child {{
          background: url("https://images.unsplash.com/photo-1640340434855-6084b1f4901c?q=80&w=2000&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
           opacity :0.8;
      }}
         </style>
         """,
        unsafe_allow_html=True)
    return


def plot_chart(df, title):
    
    my_stocks = df
    # for c in my_stocks.columns.values:
    #     st.line_chart(my_stocks[c] , color= c)
    # plt.title(title)
    # plt.xlabel('Date',fontsize=18)
    # plt.ylabel('Adj. Price USD ($)',fontsize=18)
    # plt.legend(my_stocks.columns.values,loc='upper left')
    # plt.show()
    fig = px.line(df, title = title)
    return fig

def plot_scatter(df,x,y,z,sharpeX, sharpeY):
    fig = px.scatter(df,x = x ,y = y, color = z)
    fig.add_trace(go.Scatter(x=sharpeX, y=sharpeY, mode = 'markers',
                         marker_symbol = 'star',
                         marker_size = 15))
    return fig


@st.cache_data
def get_data(assets,start_date , end_date):
    df = pd.DataFrame()
    yf.pdr_override()
    for stock in assets:
        df[stock] = web.get_data_yahoo(stock ,start = start_date,end = end_date)['Adj Close']
    
    return df

@st.cache_data
def monteCarlo(assets,  num_of_portfolios):
    log_returns = np.log(1+ df.pct_change())
    
    monte_df  = all_weights = np.zeros((num_of_portfolios, len(assets)))

    # preppping the returns array
    ret_arr = np.zeros(num_of_portfolios)

    # prepping the Volatility array
    vol_arr = np.zeros(num_of_portfolios)

    # prepping the sharpe ratio array
    sharpe_arr = np.zeros(num_of_portfolios)


    # Start the simulation

    for i in range(num_of_portfolios):

        # first calculate the weights
        monte_weights = np.array(np.random.random(len(assets)))
        monte_weights = monte_weights/np.sum(monte_weights)

        # storing the weights of each turn
        all_weights[i, :] = monte_weights

        # calculate the expected log returns
        ret_arr[i] = np.sum((log_returns.mean()*monte_weights)*252)

        # calculate the volatility,
        vol_arr[i] = np.sqrt(
            np.dot(monte_weights.T , np.dot(log_returns.cov()*252 , monte_weights))
        )

        # calculate sharpe ratio
        sharpe_arr[i] = ret_arr[i] /vol_arr[i]

    # the collection of all the data parameters we just calculated
    simulations_data = [ret_arr ,vol_arr , sharpe_arr , all_weights]

    # converting in into a dataframe
    simulations_df = pd.DataFrame(data = simulations_data).T

    # naming the columns

    simulations_df.columns = [
        "Returns",
        "Volatility",
        "Sharpe Ratio",
        "Portfolio Weights",
    ]

    
    # Make sure the data types are correct, we don't want our floats to be strings.
    simulations_df = simulations_df.infer_objects()
    return simulations_df

    
def statistics(df, weights):
    # percentage returns of the portfolio 
    returns = df.pct_change()
    # Simple annual returns 
    portfolio_simple_annual_return = np.sum(returns.mean()*weights)*252
    
    # annual covariance matrix
    cov_matrix_annual = returns.cov()*252
    
    # variance of the portfolio 
    port_variance =np.dot(weights.T ,np.dot(cov_matrix_annual , weights))
    
    port_volatility = np.sqrt(port_variance)
    
    return returns , portfolio_simple_annual_return,port_variance, port_volatility


def plot_efficient_frontier_and_max_sharpe(mu, S): 
    # Optimize portfolio for max Sharpe ratio and plot it out with efficient frontier curve
    ef = EfficientFrontier(mu, S)
    fig, ax = plt.subplots(figsize=(6,4))
    ef_max_sharpe = copy.deepcopy(ef)
    plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)
    # Find the max sharpe portfolio
    ef_max_sharpe.max_sharpe(risk_free_rate=0.02)
    ret_tangent, std_tangent, _ = ef_max_sharpe.portfolio_performance()
    ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")
    # Generate random portfolios
    n_samples = 1000
    w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)
    rets = w.dot(ef.expected_returns)
    stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
    sharpes = rets / stds
    ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")
    # Output
    ax.legend()
    return fig




try:
  ###################################################################
      # Setting up the basic page layout of the application
    st.set_page_config(page_title = "Hemant's Stock Portfolio Optimizer", 
                        page_icon="🧊",
                        layout="centered",
                        initial_sidebar_state="expanded",
                        menu_items={
                    # 'Get Help': 'https://www.extremelycoolapp.com/help',
                    # 'Report a bug': "https://www.extremelycoolapp.com/bug",
                    # 'About': "# This is a header. This is an *extremely* cool app!"
            })

    #########################################################################
    st.title("Portfolio Optimization")
    st.header("Made by Hemant ", divider = "rainbow")
    st.markdown('''
            🚀 Welcome to our Dynamic Portfolio Optimization Project! 📈

    Harnessing the power of Monte Carlo simulation and the PyPortfolioOpt package in Python 🐍, we've unlocked the secret to maximizing your portfolio's returns while skyrocketing the Sharpe ratio! 💼💰

    In a world of financial uncertainty, our innovative approach stands out:   
    📊 Monte Carlo Simulation: We simulate thousands of possible market scenarios to craft a robust and resilient portfolio strategy.  
    📈 PyPortfolioOpt Magic: Leveraging the cutting-edge capabilities of PyPortfolioOpt, we fine-tune your investments for     optimal performance.  
    💡 Maximizing Sharpe Ratio: Our focus is on achieving the highest possible risk-adjusted returns, ensuring your portfolio thrives in any market condition.

    #PortfolioOptimization #PythonMagic #MaximizeReturns
            ''')
    st.divider()
    ###############################################################
    st.sidebar.subheader("Enter the Details", divider = "rainbow")
    start_date = st.sidebar.date_input("Start Date",datetime(2013, 1, 1))
    end_date = st.sidebar.date_input("End Date") # it defaults to current date
    tickers_string = st.sidebar.text_input('Enter all stock tickers to be included in portfolio separated by commas \ WITHOUT spaces, ', '',placeholder = 'e.g. "MA,META,V,AMZN,JPM,BA"').upper()
    st.sidebar.write("Get  stock tickers [here](https://www.nasdaq.com/market-activity/stocks/screener)")
    assets = tickers_string.split(',')
    starting_amount = st.sidebar.number_input("Starting Amount ($)",value = 100, placeholder="Type a positive Amount...")
    st.sidebar.button("Enter")

    # collecting the date using yahoo finance
    df = get_data(assets, start_date=start_date, end_date=end_date) 
    weights = np.array([1/len(assets)]*len(assets))

    st.subheader("Stock Price Chart", divider="rainbow")
    price_chart = plot_chart(df, title = "")
    st.plotly_chart(price_chart)

    st.subheader("Prices of the stocks", divider =  "rainbow")
    st.markdown('''
                At this starting point, we're adopting an equal weight allocation strategy for each stock in our portfolio.💼💰 ''')
    st.dataframe(df)
    # Plotting the stocks prices
    # to override the plt function
    st.set_option('deprecation.showPyplotGlobalUse', False)
    # Printing the related stats of the portfolio
    stats = statistics(df, weights  = weights)

    # annual_returns  = np.sum(returns.mean()*weights)*252
    annual_Returns = (round(stats[1]*100,2))
    st.markdown(f"**Annual Simple Annual Return : {(round(stats[1]*100,2))}%**")
    st.markdown(f"**Annual Volatility/Risk : {round(stats[3]*100,2)}%**")
    st.markdown(f"**Annual Variance : {round(stats[2]*100,2)}%**")
    ############################################################################################################################

    st.subheader("Sharpe Ratio" , divider = "rainbow")
    st.write("""The Sharpe ratio compares the return of an investment with its risk.It's a mathematical expression of the insight that excess returns over a period of time may signify more volatility and risk, rather than investing skill.
        The Sharpe ratio is one of the most widely used methods for measuring risk-adjusted relative returns. It compares a fund's historical or projected returns relative to an investment benchmark with the historical or expected variability of such returns.

    The risk-free rate was initially used in the formula to denote an investor's hypothetical minimal borrowing costs.
    More generally, it represents the risk premium of an investment versus a safe asset such as a Treasury bill or bond.

    When benchmarked against the returns of an industry sector or investing strategy, the Sharpe ratio provides a measure of risk-adjusted performance not attributable to such affiliations.

    The ratio is useful in determining to what degree excess historical returns were accompanied by excess volatility. While excess returns are measured in comparison with an investing benchmark, the standard deviation formula gauges volatility based on the variance of returns from their mean.

    The ratio's utility relies on the assumption that the historical record of relative risk-adjusted returns has at least some predictive value.
        
            """)
    st.latex(r'''
            Sharpe\ Ratio = \frac{R_p - R_f}{\sigma_p}
            
            \\
            where:\\
            R_p  = return\ of\ protfolio\\
            R_f = risk-free\ rate\\
            \sigma_p = standard\ deviation\ of\ the\ portfolio's\ excess\ return
            
            ''')
    st.link_button("More About Sharpe Ratio", "https://www.investopedia.com/terms/s/sharperatio.asp")
    st.subheader("Choose a method: ")


    ####################################################################################################
    monte, port = st.tabs(["Monte Carlo Simulation" , "Using PyPortfolioOp"])
    with monte:
        
        st.subheader("Simulation Data", divider  =  "rainbow")
        st.write("Your Portfolio Consists of {} Stocks".format(tickers_string))
        num_of_portfolios =  st.number_input("Number of simulations", value =  100)
        sim_df = monteCarlo(assets, num_of_portfolios =num_of_portfolios )
        st.dataframe(sim_df)
        
        
        
        # Return the Max Sharpe Ratio from the run.
        max_sharpe_ratio = sim_df.loc[sim_df['Sharpe Ratio'].idxmax()]

        # Return the Min Volatility from the run.
        min_volatility = sim_df.loc[sim_df['Volatility'].idxmin()]

        # Returning the max _returns metrices
        max_returns = sim_df.loc[sim_df["Returns"].idxmax()]
        st.subheader("Representation of Monte Carlo Simulation",divider =  "rainbow")
        st.write('''
                The Monte Carlo method is a stochastic (random sampling of inputs) method to solve a statistical problem, and a simulation is a virtual representation of a problem. The Monte Carlo simulation combines the two to give us a powerful tool that allows us to obtain a distribution (array) of results for any statistical problem with numerous inputs sampled over and over again.
                * The Monte Carlo method uses a random sampling of information to solve a statistical problem; while a simulation is a way to virtually demonstrate a strategy.
                * Combined, the Monte Carlo simulation enables a user to come up with a bevy of results for a statistical problem with numerous data points sampled repeatedly.
                * The Monte Carlo simulation can be used in corporate finance, options pricing, and especially portfolio management and personal finance planning. 
                * On the downside, the simulation is limited in that it can't account for bear markets, recessions, or any other kind of financial crisis that might impact potential results.
                ''')
        monte_fig = plot_scatter(sim_df,
                                    x = sim_df["Volatility"],
                                    y = sim_df["Returns"],
                                    z =  sim_df["Sharpe Ratio"],
                                    sharpeX= [max_sharpe_ratio[1]],
                                    sharpeY=[max_sharpe_ratio[0]])
        st.plotly_chart(monte_fig)
        monte_pie = pd.DataFrame(max_returns[-1] , columns = ["Weights"] , index = assets)
        monte_pie.index.name = "Ticker"
        st.write(monte_pie)
        fig_pie_monte = px.pie(monte_pie,values = "Weights", names = monte_pie.index, title = "Percentage of each stock in portfolio" )
        st.plotly_chart(fig_pie_monte)
        st.markdown('''**Expected annual return: {}**%'''.format((max_sharpe_ratio["Returns"]*100).round(2)))
        st.markdown('**Annual volatility: {}**%'.format((max_sharpe_ratio["Volatility"]*100).round(2)))
        st.markdown('**Sharpe Ratio: {}**'.format(max_sharpe_ratio["Sharpe Ratio"].round(2)))
        
        monte_Returns = (max_sharpe_ratio["Returns"]*100).round(2)
        
        
    ######################################################################################################
    with port:
        corr_df = df.corr().round(2)

        fig_corr = px.imshow(corr_df, text_auto=True)


        # Calculate expected returns and sample covariance matrix for portfolio optimization later
        mu = expected_returns.mean_historical_return(df)
        S = risk_models.sample_cov(df)

        # Plot efficient frontier curve
        fig = plot_efficient_frontier_and_max_sharpe(mu, S)
        fig_efficient_frontier = BytesIO()
        fig.savefig(fig_efficient_frontier, format="png")

        ef = EfficientFrontier(mu, S)
        ef.max_sharpe(risk_free_rate=0.02)

        weights = ef.clean_weights()
        expected_annual_return, annual_volatility, sharpe_ratio = ef.portfolio_performance()
        weights_df = pd.DataFrame.from_dict(weights, orient = 'index')
        weights_df.columns = ['Weights']
        weights_df.index.name = "Ticker"
        # Calculate returns of portfolio with optimized weights
        df['Optimized Portfolio'] = 0
        for ticker, weight in weights.items():
            df['Optimized Portfolio'] += df[ticker]*weight

        latest_prices = get_latest_prices(df)
        da = DiscreteAllocation(weights,latest_prices,total_portfolio_value = starting_amount)

        st.subheader('Cumulative Returns of Optimized Portfolio Starting with $100', divider  = "rainbow")
        returns_optimized = plot_chart(df['Optimized Portfolio'],title = None )
        #Display everything on Streamlit
        st.write("Your Portfolio Consists of {} Stocks".format(tickers_string))	
        st.plotly_chart(returns_optimized)

        # Discrete allocation of each share per stock
        allocation,leftover = da.greedy_portfolio()
        
        
        st.subheader("Optimized Max Sharpe Portfolio Weights",divider = "rainbow")
        st.dataframe(weights_df)
        fig_pie = px.pie(weights_df,values = "Weights", names = weights_df.index, title = "Percentage of each  stock  in portfolio" )
        st.plotly_chart(fig_pie)
        st.markdown('''**Discrete allocation: {}%**'''.format(allocation))
        st.markdown('''**Funds remaining: ${:.2f}**'''.format(leftover))



        st.subheader("Optimized Max Sharpe Portfolio Performance",divider =  "rainbow")
        st.write('''
                PyPortfolioOpt is a library that implements portfolio optimization methods, including classical efficient frontier techniques and Black-Litterman allocation, as well as more recent developments in the field like shrinkage and Hierarchical Risk Parity, along with some novel experimental features like exponentially-weighted covariance matrices.

                It is extensive yet easily extensible, and can be useful for both the casual investor and the serious practitioner. Whether you are a fundamentals-oriented investor who has identified a handful of undervalued picks, or an algorithmic trader who has a basket of strategies, PyPortfolioOpt can help you combine your alpha sources in a risk-efficient way.
                ''')
        port_Returns = (expected_annual_return*100).round(2) 
        st.image(fig_efficient_frontier)
        st.write('Expected annual return: {}%'.format((expected_annual_return*100).round(2)))
        st.write('Annual volatility: {}%'.format((annual_volatility*100).round(2)))
        st.write('Sharpe Ratio: {}'.format(sharpe_ratio.round(2)))

        # fig_corr is not a plotly chart
    st.subheader("Correlation between the stocks",divider = "rainbow")
    st.write('''
                Most stocks have a correlation somewhere in the middle of the range, with a coefficient of 0 indicating no relationship whatsoever between the two securities. The stock of a software company, for example, may have little to no correlation with the stock of a chain of auto parts stores because the businesses operate so differently. For example, the auto parts store company has a much higher dependence on the movement of physical goods than the software company.
            But even such fundamental differences may not explain correlations or the lack of them. More often than not stocks show correlation of price movement because of how investors respond to them.

            For example, McDonalds (MCD) and Caterpillar (CAT) operate very different businesses, but their stocks are often highly correlated over time because both are components of major market indexes.
                ''')
    st.plotly_chart(fig_corr)
        
    st.subheader("Comparison of the returns in both of the techniques" , divider  =  "rainbow")
    hist_Data = [[annual_Returns] , [monte_Returns] , [port_Returns]]
    comparison =pd.DataFrame(hist_Data ,index = [ "Simple Annual Return" , "Monte Carlo Returns", " PyPortfolioOpt Returns"] )
    comparison.columns =  ["Returns"]
    st.markdown('''
                Here we can compare the Returns accumulated over the time for our portfolio under different optimization techniques.
                ''')
    fig_hist = px.histogram(comparison,
                            x = comparison.index ,
                            y = "Returns" ,
                            color = comparison.index,
                            text_auto = True)
    fig_hist.update_layout(yaxis_title_text = " Mean Returns",
                    xaxis_title_text = "",
    )
    st.plotly_chart(fig_hist)



except:
    st.warning('Enter correct stock tickers to be included in portfolio separated\
    by commas WITHOUT spaces, e.g. "MA,META,V,AMZN,JPM,BA"and hit Enter.')	
    


# hide_streamlit_style = """

# <style>
# #MainMenu {visibility: hidden;}
# footer {visibility: hidden;}
# </style>
# """

bg_png("https://unsplash.com/photos/person-holding-light-bulb-fIq0tET6llw")