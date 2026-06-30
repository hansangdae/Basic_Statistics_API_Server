# api.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
import json

app = FastAPI(title="Statistics Learning API")

# CORS 설정 (모바일 앱에서 요청을 받을 수 있게)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 탭 1: 데이터 생성 엔드포인트
# ==========================================
@app.post("/api/generate-data")
def generate_data(
    dist_type: str,
    dist_choice: str,
    params: dict,
    n_samples: int
):
    """
    확률분포에서 샘플 데이터를 생성합니다.
    
    예: 
    {
        "dist_type": "연속분포 (Continuous)",
        "dist_choice": "정규분포 (Normal)",
        "params": {"평균 (μ)": 0, "표준편차 (σ)": 1},
        "n_samples": 100
    }
    """
    try:
        np.random.seed()
        
        if "정규분포" in dist_choice:
            data = np.random.normal(
                params["평균 (μ)"], 
                params["표준편차 (σ)"], 
                n_samples
            )
        elif "지수분포" in dist_choice:
            data = np.random.exponential(
                1/params["비율 모수 (λ)"], 
                n_samples
            )
        elif "이항분포" in dist_choice:
            data = np.random.binomial(
                int(params["시행 횟수 (n)"]), 
                params["성공 확률 (p)"], 
                n_samples
            )
        elif "포아송" in dist_choice:
            data = np.random.poisson(
                params["발생률 (λ)"], 
                n_samples
            )
        elif "t-분포" in dist_choice:
            data = np.random.standard_t(
                int(params["자유도 (df)"]), 
                n_samples
            )
        elif "카이제곱" in dist_choice:
            data = np.random.chisquare(
                int(params["자유도 (df)"]), 
                n_samples
            )
        else:
            return {"error": "Unknown distribution"}
        
        return {
            "success": True,
            "data": data.tolist(),
            "count": len(data)
        }
    except Exception as e:
        return {"error": str(e)}

# ==========================================
# 탭 2: 기술통계량 계산 엔드포인트
# ==========================================
@app.post("/api/descriptive-stats")
def descriptive_stats(data: list):
    """
    입력된 데이터의 기술통계량을 계산하고 히스토그램을 그립니다.
    """
    try:
        data = np.array(data)
        data = data[np.isfinite(data)]
        
        if len(data) < 2:
            return {"error": "At least 2 data points required"}
        
        # 통계량 계산
        stats_result = {
            "N": len(data),
            "Mean": float(np.mean(data)),
            "Variance": float(np.var(data, ddof=1)),
            "Std": float(np.std(data, ddof=1)),
            "Median": float(np.median(data)),
            "Min": float(np.min(data)),
            "Max": float(np.max(data)),
            "Range": float(np.max(data) - np.min(data)),
            "Q1": float(np.percentile(data, 25)),
            "Q3": float(np.percentile(data, 75)),
        }
        
        # 히스토그램 이미지 생성
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(data, bins=30, kde=True, stat="density", color='skyblue', ax=ax)
        ax.set_title("Histogram & Density Plot")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        
        # Base64로 변환
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor='white')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        
        return {
            "success": True,
            "stats": stats_result,
            "chart": img_base64
        }
    except Exception as e:
        return {"error": str(e)}

# ==========================================
# 탭 3: 확률분포 그래프 엔드포인트
# ==========================================
@app.post("/api/probability-distribution")
def probability_distribution(
    dist_choice: str,
    params: dict,
    func_type: str = "PDF"
):
    """
    이론적 확률분포 그래프를 그립니다.
    
    func_type: "PDF" 또는 "CDF"
    """
    try:
        # 도메인 범위 설정
        if "정규분포" in dist_choice:
            mu, sigma = params["평균 (μ)"], params["표준편차 (σ)"]
            x = np.linspace(mu - 4*sigma, mu + 4*sigma, 500)
            y = stats.norm.pdf(x, mu, sigma) if func_type == "PDF" else stats.norm.cdf(x, mu, sigma)
            
        elif "지수분포" in dist_choice:
            lam = params["비율 모수 (λ)"]
            x = np.linspace(0, 10/lam, 500)
            y = stats.expon.pdf(x, scale=1/lam) if func_type == "PDF" else stats.expon.cdf(x, scale=1/lam)
            
        elif "t-분포" in dist_choice:
            df_val = int(params["자유도 (df)"])
            x = np.linspace(-5, 5, 500)
            y = stats.t.pdf(x, df_val) if func_type == "PDF" else stats.t.cdf(x, df_val)
            
        elif "카이제곱" in dist_choice:
            df_val = int(params["자유도 (df)"])
            x = np.linspace(0, max(20, df_val*3), 500)
            y = stats.chi2.pdf(x, df_val) if func_type == "PDF" else stats.chi2.cdf(x, df_val)
            
        elif "포아송" in dist_choice:
            lam = params["발생률 (λ)"]
            x = np.arange(0, int(lam*3)+10)
            y = stats.poisson.pmf(x, lam) if func_type == "PDF" else stats.poisson.cdf(x, lam)
            
        elif "이항분포" in dist_choice:
            n_val = int(params["시행 횟수 (n)"])
            p_val = params["성공 확률 (p)"]
            x = np.arange(0, n_val+1)
            y = stats.binom.pmf(x, n_val, p_val) if func_type == "PDF" else stats.binom.cdf(x, n_val, p_val)
        else:
            return {"error": "Unknown distribution"}
        
        # 그래프 생성
        fig, ax = plt.subplots(figsize=(8, 5))
        if "이항" in dist_choice or "포아송" in dist_choice:
            ax.bar(x, y, color='#2E86AB', alpha=0.7, width=0.6)
        else:
            ax.plot(x, y, color='#2E86AB', linewidth=2.5)
            ax.fill_between(x, y, alpha=0.15, color='#2E86AB')
        
        ax.set_title(f"{dist_choice} — {func_type}")
        ax.set_xlabel("X")
        ax.set_ylabel("Probability")
        ax.grid(True, linestyle='--', alpha=0.4)
        
        # Base64로 변환
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor='white')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        
        return {
            "success": True,
            "chart": img_base64
        }
    except Exception as e:
        return {"error": str(e)}

# ==========================================
# 탭 4: 가설검정 & 추정 엔드포인트
# ==========================================
@app.post("/api/hypothesis-test")
def hypothesis_test(
    data: list,
    h0_value: float,
    analysis_type: str,
    test_direction: str,
    confidence_level: float
):
    """
    모평균 또는 모분산 검정 및 신뢰구간 추정
    
    analysis_type: "mean" 또는 "variance"
    test_direction: "two", "right", "left"
    """
    try:
        data = np.array(data)
        data = data[np.isfinite(data)]
        
        if len(data) < 2:
            return {"error": "Insufficient data"}
        
        n = len(data)
        alpha = 1 - (confidence_level / 100)
        sample_mean = np.mean(data)
        sample_var = np.var(data, ddof=1)
        sample_std = np.std(data, ddof=1)
        
        if analysis_type == "mean":
            se = np.sqrt(sample_var / n)
            t_stat = (sample_mean - h0_value) / se
            df_val = n - 1
            
            if test_direction == "two":
                p_val = stats.t.sf(np.abs(t_stat), df_val) * 2
                t_crit = stats.t.ppf(1 - alpha/2, df_val)
            elif test_direction == "right":
                p_val = stats.t.sf(t_stat, df_val)
                t_crit = stats.t.isf(alpha, df_val)
            else:  # left
                p_val = stats.t.cdf(t_stat, df_val)
                t_crit = stats.t.ppf(alpha, df_val)
            
            margin = t_crit * se
            ci_lower = sample_mean - margin
            ci_upper = sample_mean + margin
            
            # 그래프
            fig, ax = plt.subplots(figsize=(8, 4))
            x_plot = np.linspace(-5, 5, 500)
            y_plot = stats.t.pdf(x_plot, df_val)
            ax.plot(x_plot, y_plot, color='#2E86AB', linewidth=2)
            ax.axvline(t_stat, color='#E74C3C', linewidth=2, linestyle='--', label=f't={t_stat:.2f}')
            ax.set_title("t-Test Distribution")
            ax.set_xlabel("Test Statistic")
            ax.set_ylabel("Density")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.4)
            
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", facecolor='white')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode("utf-8")
            plt.close(fig)
            
            return {
                "success": True,
                "test_stat": float(t_stat),
                "p_value": float(p_val),
                "ci_lower": float(ci_lower),
                "ci_upper": float(ci_upper),
                "rejected": bool(p_val < alpha),
                "chart": img_base64
            }
        
        else:  # variance
            df_val = n - 1
            chi2_stat = (df_val * sample_var) / h0_value
            
            if test_direction == "two":
                p_val = 2 * min(stats.chi2.cdf(chi2_stat, df_val), stats.chi2.sf(chi2_stat, df_val))
            elif test_direction == "right":
                p_val = stats.chi2.sf(chi2_stat, df_val)
            else:
                p_val = stats.chi2.cdf(chi2_stat, df_val)
            
            chi2_low = stats.chi2.ppf(alpha/2, df_val)
            chi2_high = stats.chi2.isf(alpha/2, df_val)
            ci_lower = (df_val * sample_var) / chi2_high
            ci_upper = (df_val * sample_var) / chi2_low
            
            # 그래프
            fig, ax = plt.subplots(figsize=(8, 4))
            x_plot = np.linspace(0, max(chi2_stat * 1.5, df_val * 2.5), 500)
            y_plot = stats.chi2.pdf(x_plot, df_val)
            ax.plot(x_plot, y_plot, color='#2E86AB', linewidth=2)
            ax.axvline(chi2_stat, color='#E74C3C', linewidth=2, linestyle='--', label=f'χ²={chi2_stat:.2f}')
            ax.set_title("Chi-Square Distribution")
            ax.set_xlabel("Test Statistic")
            ax.set_ylabel("Density")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.4)
            
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", facecolor='white')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode("utf-8")
            plt.close(fig)
            
            return {
                "success": True,
                "test_stat": float(chi2_stat),
                "p_value": float(p_val),
                "ci_lower": float(ci_lower),
                "ci_upper": float(ci_upper),
                "rejected": bool(p_val < alpha),
                "chart": img_base64
            }
    
    except Exception as e:
        return {"error": str(e)}

# ==========================================
# 탭 5: 회귀분석 엔드포인트
# ==========================================
@app.post("/api/regression-analysis")
def regression_analysis(
    x_values: list,
    y_values: list
):
    """
    선형 회귀분석을 수행합니다.
    """
    try:
        x = np.array(x_values)
        y = np.array(y_values)
        
        res = stats.linregress(x, y)
        slope = res.slope
        intercept = res.intercept
        r_value = res.rvalue
        r_squared = r_value ** 2
        p_value = res.pvalue
        
        # 그래프
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(x, y, color='#1f77b4', alpha=0.7, s=30, label='Data')
        
        x_line = np.linspace(np.min(x), np.max(x), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color='#d62728', linewidth=2.5, label=f'Regression: Y={slope:.2f}X+{intercept:.2f}')
        
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Linear Regression")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.4)
        
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor='white')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        
        return {
            "success": True,
            "slope": float(slope),
            "intercept": float(intercept),
            "r_value": float(r_value),
            "r_squared": float(r_squared),
            "p_value": float(p_value),
            "chart": img_base64
        }
    except Exception as e:
        return {"error": str(e)}

# ==========================================
# 헬스 체크
# ==========================================
@app.get("/")
def read_root():
    return {
        "message": "Statistics Learning API Server is running!",
        "version": "1.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
