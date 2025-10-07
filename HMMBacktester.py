import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GMMHMM
import pandas_datareader.data as web
plt.style.use('dark_background')


class HMMBacktester:
    def __init__(
        self: "HMMBacktester",
        start: dt.datetime,
        end: dt.datetime,
        asset_names: list[str],
        window_size: int = 360,
        n_components: int = 2,
    ):
        self.start = start
        self.end = end
        self.asset_names = asset_names
        self.window_size = window_size
        self.n_components = n_components
        self.W_safe_asset_list = []
        self.strategy_returns = []

    def download_data(self: "HMMBacktester"):
        safe_asset = web.DataReader(
            f"{self.asset_names[1]}.US", "stooq", self.start, self.end
        )
        risky_asset = web.DataReader(
            f"{self.asset_names[0]}.US", "stooq", self.start, self.end
        )
        safe_asset = safe_asset.sort_index(ascending=True)
        risky_asset = risky_asset.sort_index(ascending=True)
        risky_asset_ret = pd.DataFrame(
            np.log(risky_asset["Close"] / risky_asset["Close"].shift(1)),
            index=risky_asset.index,
        ).dropna()
        safe_asset_ret = pd.DataFrame(
            np.log(safe_asset["Close"] / safe_asset["Close"].shift(1)),
            index=safe_asset.index,
        ).dropna()
        total_ret = pd.merge(
            risky_asset_ret,
            safe_asset_ret,
            left_index=True,
            right_index=True,
            how="inner",
        )
        total_ret.columns = [
            self.asset_names[0] + "_logRet",
            self.asset_names[1] + "_logRet",
        ]
        self.total_ret = total_ret

    def run(self: "HMMBacktester"):
        for i in range(self.window_size, len(self.total_ret) - 1):
            train = self.total_ret.iloc[i - self.window_size : i]
            test_idx = self.total_ret.index[i]
            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(
                train[[self.asset_names[0] + "_logRet"]]
            )
            model = GMMHMM(
                n_components=self.n_components,
                n_mix=3,
                covariance_type="tied",
                n_iter=100,
                random_state=0,
            )
            model.fit(train_scaled)
            post = model.predict_proba(train_scaled)
            pi_next = post[-1] @ model.transmat_
            train = train.copy()
            train["State"] = model.predict(train_scaled)
            state_ret = train.groupby("State")[self.asset_names[0] + "_logRet"].mean()
            high_risk_state = state_ret.idxmin()
            W_safe_asset = pi_next[high_risk_state]
            W_risky_asset = 1 - W_safe_asset
            self.W_safe_asset_list.append(W_safe_asset)
            ret_risky_asset = self.total_ret.loc[
                test_idx, self.asset_names[0] + "_logRet"
            ]
            ret_safe_asset = self.total_ret.loc[
                test_idx, self.asset_names[1] + "_logRet"
            ]
            strategy_ret = np.log(
                W_safe_asset * np.exp(ret_safe_asset)
                + W_risky_asset * np.exp(ret_risky_asset)
            )
            self.strategy_returns.append(
                (test_idx, strategy_ret, ret_risky_asset, ret_safe_asset)
            )
        self.strategy_returns = pd.DataFrame(
            self.strategy_returns,
            columns=[
                "Date",
                "Strategy Returns",
                self.asset_names[0] + "_logRet",
                self.asset_names[1] + "_logRet",
            ],
        )
        self.strategy_returns.set_index("Date", inplace=True)

    def plot(self: "HMMBacktester") -> None:
        strategy_pl = np.exp(self.strategy_returns["Strategy Returns"].cumsum())
        risky_asset_pl = np.exp(
            self.strategy_returns[self.asset_names[0] + "_logRet"].cumsum()
        )
        safe_asset_pl = np.exp(
            self.strategy_returns[self.asset_names[1] + "_logRet"].cumsum()
        )
        fig, ax1 = plt.subplots(figsize=(10, 5))
        steps = list(range(len(self.W_safe_asset_list)))
        fill = ax1.fill_between(
            self.strategy_returns.index[-len(steps) :],
            self.W_safe_asset_list,
            color="gold",
            alpha=0.4,
            label=r"$W_{\mathrm{Safe}}$",
        )
        ax1.set_xlabel("Step")
        ax1.set_ylabel(r"$W_{\mathrm{Safe}}$", color="gold")
        ax1.tick_params(axis="y", labelcolor="gold")
        ax1.grid(color="gray", linestyle=":")
        ax2 = ax1.twinx()
        line1 = ax2.plot(
            self.strategy_returns.index[-len(steps) :],
            risky_asset_pl,
            color="deepskyblue",
            label=f"{self.asset_names[0]} PL",
            linewidth=2,
        )[0]
        line2 = ax2.plot(
            self.strategy_returns.index[-len(steps) :],
            strategy_pl,
            color="red",
            label="HMM Strategy PL",
            linewidth=2,
        )[0]
        ax2.set_ylabel(f"{self.asset_names[0]} PL", color="deepskyblue")
        ax2.tick_params(axis="y", labelcolor="deepskyblue")
        handles = [fill, line1, line2]
        labels = [h.get_label() for h in handles]
        ax1.legend(handles, labels, loc="upper left")
        fig.tight_layout()
        plt.show()


start = dt.datetime.now() - dt.timedelta(days=3 * 365)
end = dt.datetime.now()
asset_names = ["QQQ", "GLD"]

bt = HMMBacktester(start, end, asset_names)
bt.download_data()
bt.run()
bt.plot()
