import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generates visual and text evidence for the KRNN Risk Pipeline.
    """

    def __init__(self, output_dir="./reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        try:
            import seaborn as sns
            sns.set_theme(style="whitegrid")
        except ImportError:
            pass

    def plot_regime_clustering(self, mus, sigmas):
        """Visual Evidence of 'Mean Collapse'."""
        plt.figure(figsize=(10, 6))
        plt.scatter(mus, sigmas, alpha=0.6, edgecolors='w', s=50)

        mu_mean = np.mean(mus)
        sigma_mean = np.mean(sigmas)

        plt.axvline(mu_mean, color='r', linestyle='--', alpha=0.5, label=f'Avg Mu: {mu_mean:.4f}')
        plt.axhline(sigma_mean, color='g', linestyle='--', alpha=0.5, label=f'Avg Vol: {sigma_mean:.4f}')

        plt.title("Evidence of Regime Identification (Mean Collapse)")
        plt.xlabel("Predicted Mean Return (Mu)")
        plt.ylabel("Predicted Volatility (Sigma)")
        plt.legend()
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, "1_regime_clustering.png")
        plt.savefig(save_path)
        plt.close()

    def plot_qq_residuals(self, residuals, ticker_name):
        """
        Generates a Q-Q plot to visually confirm non-normality.
        """
        plt.figure(figsize=(10, 6))

        # Compare residuals against Standard Normal
        stats.probplot(residuals, dist="norm", plot=plt)

        plt.title(f"Q-Q Plot: {ticker_name} Residuals vs Normal Distribution")
        plt.xlabel("Theoretical Quantiles (Gaussian)")
        plt.ylabel("Observed Quantiles (Standardized Residuals)")
        plt.grid(True, alpha=0.3)

        save_path = os.path.join(self.output_dir, f"qq_{ticker_name}.png")
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Saved Q-Q plot for {ticker_name} to {save_path}")

    def plot_tail_comparison(self, tail_data):
        """Visual Evidence of Risk Differentiation."""
        sorted_tickers = sorted(tail_data.keys(), key=lambda k: tail_data[k]['gamma'])
        safe_ticker = sorted_tickers[0]
        risky_ticker = sorted_tickers[-1]

        safe_gamma = tail_data[safe_ticker]['gamma']
        risky_gamma = tail_data[risky_ticker]['gamma']

        safe_resid = tail_data[safe_ticker]['residuals']
        risky_resid = tail_data[risky_ticker]['residuals']

        plt.figure(figsize=(12, 6))
        plt.hist(safe_resid, bins=50, density=True, alpha=0.5, label=f"{safe_ticker} (Gamma={safe_gamma:.2f})",
                 color='blue')
        plt.hist(risky_resid, bins=50, density=True, alpha=0.5, label=f"{risky_ticker} (Gamma={risky_gamma:.2f})",
                 color='red')

        plt.title(f"Tail Risk Differentiation: {safe_ticker} vs {risky_ticker}")
        plt.xlabel("Standardized Residuals (Z-Scores)")
        plt.ylabel("Density")
        plt.legend()
        plt.xlim(-5, 5)
        plt.figtext(0.15, 0.8, "Note: Higher Gamma implies fatter left tail (Crash Risk)", fontsize=10)

        save_path = os.path.join(self.output_dir, "2_tail_differentiation.png")
        plt.savefig(save_path)
        plt.close()

    def plot_allocation_vs_risk(self, tickers, weights, gammas):
        """Visual Evidence of Optimization Logic."""
        df = pd.DataFrame({'Ticker': tickers, 'Weight': weights, 'Gamma': gammas})

        plt.figure(figsize=(10, 6))
        plt.scatter(df['Gamma'], df['Weight'], s=100, c=df['Weight'], cmap='viridis')

        for i, row in df.iterrows():
            plt.text(row['Gamma'], row['Weight'], row['Ticker'], fontsize=9)

        if len(df) > 1:
            z = np.polyfit(df['Gamma'], df['Weight'], 1)
            p = np.poly1d(z)
            plt.plot(df['Gamma'], p(df['Gamma']), "r--", alpha=0.5, label="Trend (Higher Risk -> Lower Weight)")

        plt.title("Optimization Logic: Impact of Tail Risk on Capital Allocation")
        plt.xlabel("Tail Index (Gamma)")
        plt.ylabel("Optimal Portfolio Weight")
        plt.legend()
        plt.grid(True, alpha=0.3)

        save_path = os.path.join(self.output_dir, "3_allocation_logic.png")
        plt.savefig(save_path)
        plt.close()

    def generate_diagnostics(self, preds, targets):
        """Out-of-Sample Performance Visualization."""
        # R-Squared
        ss_res = np.sum((targets - preds) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        plt.figure(figsize=(10, 6))
        plt.scatter(targets, preds, alpha=0.1)
        min_val = min(targets.min(), preds.min())
        max_val = max(targets.max(), preds.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Perfect Prediction")

        plt.title(f"Prediction Accuracy (Test Set R2: {r2:.4f})")
        plt.xlabel("Actual Return")
        plt.ylabel("Predicted Return")
        plt.legend()

        save_path = os.path.join(self.output_dir, "4_prediction_accuracy.png")
        plt.savefig(save_path)
        plt.close()
        return r2

    def plot_risk_bounds_comparison(self, bounds_data: list):
        """
        Visualizes the gap between EVT estimates and Worst-Case Robust Bounds.
        bounds_data: List of dicts with {Ticker, EVT_CVaR, DMP_CVaR, Gamma}
        """
        if not bounds_data:
            return

        tickers = [x['Ticker'] for x in bounds_data]
        evt_vals = [x['EVT_CVaR'] for x in bounds_data]
        dmp_vals = [x['DMP_CVaR'] for x in bounds_data]

        x = np.arange(len(tickers))
        width = 0.35

        plt.figure(figsize=(12, 6))
        plt.bar(x - width / 2, evt_vals, width, label='EVT Estimate (Historical)', color='skyblue', alpha=0.8)
        plt.bar(x + width / 2, dmp_vals, width, label='DMP Bound (Worst-Case)', color='salmon', alpha=0.8)

        plt.ylabel('CVaR (Z-Score units)')
        plt.title('Risk Ambiguity: Estimated vs. Worst-Case CVaR')
        plt.xticks(x, tickers, rotation=45)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.3)

        # Annotate the "Risk Gap"
        for i in range(len(tickers)):
            gap = dmp_vals[i] - evt_vals[i]
            plt.text(i, max(dmp_vals[i], evt_vals[i]) + 0.1, f"+{gap:.2f}", ha='center', fontsize=8, color='red')

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, "5_risk_bounds_gap.png")
        plt.savefig(save_path)
        plt.close()

    # Update save_comprehensive_report to accept bounds_data and print columns
    def save_comprehensive_report(self, candidates, portfolio, opt_metrics, test_metrics, bounds_data=None):
        path = os.path.join(self.output_dir, "comprehensive_summary.txt")
        with open(path, "w") as f:
            f.write("=== KRNN RISK MANAGEMENT PIPELINE REPORT ===\n")
            f.write("============================================\n\n")

            f.write("--- PHASE 3: SYSTEMATIC SELECTION & ROBUST RISK ANALYSIS ---\n")
            f.write("Note: WC-VaR/WC-CVaR are Worst-Case bounds derived via Linear Programming (Discrete Moment Problem).\n")
            f.write(
                f"{'Ticker':<8} {'Mu(%)':<8} {'Vol(%)':<8} {'Gamma':<8} {'EVT-VaR':<10} {'WC-VaR':<10} {'EVT-CVaR':<10} {'WC-CVaR':<10} {'Risk Gap':<10}\n")
            f.write("-" * 110 + "\n")

            # Map bounds data for easy access (CVaR)
            dmp_map = {x['Ticker']: x['DMP_CVaR'] for x in (bounds_data or [])}

            for c in candidates:
                dmp_val = dmp_map.get(c['Ticker'], 0.0)
                evt_var = c.get('VaR', 0.0)
                wc_var = c.get('WC_VaR', 0.0)
                evt_cvar = c.get('ES', 0.0)
                risk_gap = dmp_val - evt_cvar
                f.write(
                    f"{c['Ticker']:<8} "
                    f"{c['Mu'] * 100:<8.2f} "
                    f"{c['Sigma'] * 100:<8.2f} "
                    f"{c['Gamma']:<8.2f} "
                    f"{evt_var:<10.2f} "
                    f"{wc_var:<10.2f} "
                    f"{evt_cvar:<10.2f} "
                    f"{dmp_val:<10.2f} "
                    f"{risk_gap:<10.2f}\n"
                )
            f.write("\n")

            f.write("--- PHASE 4: OPTIMIZATION RESULTS ---\n")
            f.write(f"Objective: Minimize CVaR (95%)\n")
            f.write(f"Target Daily Return: > {opt_metrics.get('Target_Ret', 0) * 100:.4f}%\n")
            f.write(f"Optimized Portfolio CVaR: {opt_metrics.get('CVaR', 0):.4f}\n\n")
            f.write("Final Portfolio Allocation:\n")
            f.write(f"{'Ticker':<8} {'Weight (%)':<12} {'Tail Risk (Gamma)':<20}\n")
            f.write("-" * 50 + "\n")
            # Combine portfolio data for writing
            # portfolio is list of dicts: {'Ticker': t, 'Weight': w, 'Gamma': g}
            for p in portfolio:
                f.write(f"{p['Ticker']:<8} {p['Weight'] * 100:<12.2f} {p['Gamma']:<20.4f}\n")
            f.write("\n")

            f.write("--- PHASE 5: OUT-OF-SAMPLE DIAGNOSTICS (2024+) ---\n")
            f.write(f"Test Set R-Squared: {test_metrics.get('R2', 0):.4f}\n")
            f.write(f"Cumulative Market Return:   {test_metrics.get('Market_Cum', 0) * 100:.2f}%\n")
            f.write(f"Cumulative Strategy Return: {test_metrics.get('Strategy_Cum', 0) * 100:.2f}%\n")

        logger.info(f"Saved comprehensive text report to {path}")
