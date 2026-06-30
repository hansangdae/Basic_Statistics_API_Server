# api.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use("agg") # 서버 환경에서 그래프를 그리기 위한 필수 설정
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io

app = FastAPI(title="Statistics API Server")

# 모바일 앱에서 접속할 수 있도록 CORS 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 모델 정의
class GenerateDataRequest(BaseModel):
    dist_choice: str
    params: dict
    n_samples: int

class DescriptiveRequest(BaseModel):
    data: list
    bins: int
    bar_color: str
    curve_color: str

class DistGraphRequest(BaseModel):
    dist_choice: str
    params: dict
    func_type: str
    line_color: str

class HypothesisRequest(BaseModel):
    data: list
    h0_val: float
    analysis_mode: str
    test_type: str
    alpha: float

class RegressionRequest(BaseModel):
    a: float
    b: float
    sigma: float
    n: int
    point_color: str  # 추가됨
    line_color: str   # 추가됨

# 공통 함수: 그래프를 Base64 이미지로 변환
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor='white')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_base64

@app.get("/")
def read_root():
    return {"status": "API Server is Running!"}

# 1단계: 데이터 생성
@app.post("/api/generate")
def generate_data(req: GenerateDataRequest):
    try:
        np.random.seed()
        n = req.n_samples
        p = req.params
        c = req.dist_choice
        
        if "정규분포" in c: data = np.random.normal(p["평균 (μ)"], p["표준편차 (σ)"], n)
        elif "지수분포" in c: data = np.random.exponential(1/p["비율 모수 (λ)"], n)
        elif "이항분포" in c: data = np.random.binomial(int(p["시행 횟수 (n)"]), p["성공 확률 (p)"], n)
        elif "포아송" in c: data = np.random.poisson(p["평균 발생률 (λ)"], n)
        elif "t-분포" in c: data = np.random.standard_t(int(p["자유도 (df)"]), n)
        elif "카이제곱" in c: data = np.random.chisquare(int(p["자유도 (df)"]), n)
        else: return {"error": "알 수 없는 분포입니다."}
        
        return {"success": True, "data": data.tolist()}
    except Exception as e:
        return {"error": str(e)}

# 2단계: 기술통계량 및 그래프
@app.post("/api/descriptive")
def descriptive_stats(req: DescriptiveRequest):
    try:
        data = np.array(req.data)
        if len(data) < 2: return {"error": "데이터가 2개 이상 필요합니다."}
        
        stats_dict = {
            "데이터 수 (N)": len(data),
            "평균 (Mean)": float(np.mean(data)),
            "분산 (Variance)": float(np.var(data, ddof=1)),
            "표준편차 (Std)": float(np.std(data, ddof=1)),
            "중앙값 (Median)": float(np.median(data)),
            "최솟값 (Min)": float(np.min(data)),
            "최댓값 (Max)": float(np.max(data)),
            "범위 (Range)": float(np.max(data) - np.min(data)),
            "Q1 (25%)": float(np.percentile(data, 25)),
            "Q3 (75%)": float(np.percentile(data, 75)),
        }
        
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.histplot(data, bins=req.bins, kde=False, stat="density", color=req.bar_color, alpha=0.6, ax=ax)
        sns.kdeplot(data, color=req.curve_color, linewidth=2, ax=ax)
        ax.set_title("Histogram & Density (KDE)", fontsize=12, color='#1E3A5F')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle='--', alpha=0.4)
        
        return {"success": True, "stats": stats_dict, "image": fig_to_base64(fig)}
    except Exception as e:
        return {"error": str(e)}

# 3단계: 확률분포 그래프
@app.post("/api/distribution")
def distribution_graph(req: DistGraphRequest):
    try:
        fig, ax = plt.subplots(figsize=(5, 3))
        c = req.dist_choice
        p = req.params
        color = req.line_color
        is_pdf = (req.func_type == "PDF")
        
        if "정규분포" in c:
            mu, sigma = p["평균 (μ)"], p["표준편차 (σ)"]
            x = np.linspace(mu - 4*sigma, mu + 4*sigma, 500)
            y = stats.norm.pdf(x, mu, sigma) if is_pdf else stats.norm.cdf(x, mu, sigma)
            ax.plot(x, y, color=color, linewidth=2.5)
            ax.fill_between(x, y, alpha=0.15, color=color)
        elif "지수분포" in c:
            lam = p["비율 모수 (λ)"]
            x = np.linspace(0, 10/lam, 500)
            y = stats.expon.pdf(x, scale=1/lam) if is_pdf else stats.expon.cdf(x, scale=1/lam)
            ax.plot(x, y, color=color, linewidth=2.5)
            ax.fill_between(x, y, alpha=0.15, color=color)
        elif "t-분포" in c:
            df_val = p["자유도 (df)"]
            x = np.linspace(-5, 5, 500)
            y = stats.t.pdf(x, df_val) if is_pdf else stats.t.cdf(x, df_val)
            ax.plot(x, y, color=color, linewidth=2.5)
            ax.fill_between(x, y, alpha=0.15, color=color)
        elif "카이제곱" in c:
            df_val = p["자유도 (df)"]
            x = np.linspace(0, max(20, df_val*3), 500)
            y = stats.chi2.pdf(x, df_val) if is_pdf else stats.chi2.cdf(x, df_val)
            ax.plot(x, y, color=color, linewidth=2.5)
            ax.fill_between(x, y, alpha=0.15, color=color)
        elif "포아송" in c:
            lam = p["평균 발생률 (λ)"]
            x = np.arange(0, int(lam*3)+10)
            y = stats.poisson.pmf(x, lam) if is_pdf else stats.poisson.cdf(x, lam)
            ax.bar(x, y, color=color, alpha=0.7, width=0.6)
        elif "이항분포" in c:
            n_val, p_val = int(p["시행 횟수 (n)"]), p["성공 확률 (p)"]
            x = np.arange(0, n_val+1)
            y = stats.binom.pmf(x, n_val, p_val) if is_pdf else stats.binom.cdf(x, n_val, p_val)
            ax.bar(x, y, color=color, alpha=0.7, width=0.6)
            
        ax.set_title(f"{c} - {req.func_type}", fontsize=11, color='#1E3A5F')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle='--', alpha=0.4)
        
        return {"success": True, "image": fig_to_base64(fig)}
    except Exception as e:
        return {"error": str(e)}

# 4단계: 가설검정
@app.post("/api/hypothesis")
def hypothesis_test(req: HypothesisRequest):
    try:
        data = np.array(req.data)
        n = len(data)
        if n < 2: return {"error": "데이터가 부족합니다."}
        
        mean = np.mean(data)
        var = np.var(data, ddof=1)
        std = np.std(data, ddof=1)
        
        fig, ax = plt.subplots(figsize=(5, 3))
        result_dict = {}
        
        if req.analysis_mode == "test":
            se = np.sqrt(var / n)
            t_stat = (mean - req.h0_val) / se
            df_val = n - 1
            
            if req.test_type == "two": p_val = stats.t.sf(abs(t_stat), df_val) * 2
            elif req.test_type == "right": p_val = stats.t.sf(t_stat, df_val)
            else: p_val = stats.t.cdf(t_stat, df_val)
            
            t_crit = stats.t.ppf(1 - req.alpha/2, df_val)
            ci_low = mean - t_crit * se
            ci_high = mean + t_crit * se
            
            result_dict = {
                "t_stat": float(t_stat), "p_val": float(p_val),
                "ci_low": float(ci_low), "ci_high": float(ci_high),
                "rejected": bool(p_val < req.alpha)
            }
            
            x_plot = np.linspace(-5, 5, 500)
            y_plot = stats.t.pdf(x_plot, df_val)
            ax.plot(x_plot, y_plot, color='#2E86AB', linewidth=2)
            ax.axvline(t_stat, color='#E74C3C', linewidth=2, linestyle='--')
            ax.set_title("t-Test", fontsize=11)
        else:
            df_val = n - 1
            t_crit = stats.t.ppf(1 - req.alpha/2, df_val)
            se_mean = std / np.sqrt(n)
            chi2_low = stats.chi2.ppf(req.alpha/2, df_val)
            chi2_high = stats.chi2.ppf(1 - req.alpha/2, df_val)
            
            result_dict = {
                "mean": float(mean), "std": float(std),
                "mean_ci_low": float(mean - t_crit * se_mean),
                "mean_ci_high": float(mean + t_crit * se_mean),
                "std_ci_low": float(np.sqrt(df_val * var / chi2_high)),
                "std_ci_high": float(np.sqrt(df_val * var / chi2_low))
            }
            
            x_plot = np.linspace(mean - 4*std, mean + 4*std, 500)
            y_plot = stats.norm.pdf(x_plot, mean, std)
            ax.plot(x_plot, y_plot, color='#2E86AB', linewidth=2)
            ax.fill_between(x_plot, y_plot, alpha=0.15, color='#2E86AB')
            ax.set_title("Estimation", fontsize=11)
            
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        return {"success": True, "results": result_dict, "image": fig_to_base64(fig)}
    except Exception as e:
        return {"error": str(e)}

# 5단계: 회귀분석
@app.post("/api/regression")
def regression_analysis(req: RegressionRequest):
    try:
        x = np.random.uniform(0, 10, req.n)
        y = req.a * x + req.b + np.random.normal(0, req.sigma, req.n)
        
        res = stats.linregress(x, y)
        
        fig, ax = plt.subplots(figsize=(5, 3))
        # 색상 변수 적용
        ax.scatter(x, y, color=req.point_color, alpha=0.6, s=30, edgecolors='none')
        x_line = np.linspace(np.min(x), np.max(x), 100)
        ax.plot(x_line, res.slope * x_line + res.intercept, color=req.line_color, linewidth=2.5)
        ax.set_title("Linear Regression", fontsize=11)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        return {
            "success": True,
            "slope": float(res.slope), "intercept": float(res.intercept),
            "r": float(res.rvalue), "r2": float(res.rvalue**2), "p": float(res.pvalue),
            "image": fig_to_base64(fig)
        }
    except Exception as e:
        return {"error": str(e)}
