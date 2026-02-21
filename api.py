import pickle
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scipy.optimize import minimize
import os

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")
data = pickle.load(open(model_path,"rb"))
model = data["model"]
le_product = data["le_product"]
le_season = data["le_season"]

class PricingRequest(BaseModel):
    product: str
    season: str
    marketing: float
    competitor_price: float
    cost: float
    min_price: float
    max_price: float

def predict_demand(price, product, season, marketing, competitor_price):
    pe = le_product.transform([product])[0]
    se = le_season.transform([season])[0]
    pr = price / competitor_price
    ml = np.log1p(marketing)
    return float(max(0, model.predict([[pe, price, marketing, competitor_price, se, pr, ml]])[0]))

@app.get("/")
def root():
    return {"status": "AI Pricing Engine Running!"}

@app.post("/optimize")
def optimize(req: PricingRequest):
    def neg_profit(price):
        d = predict_demand(price[0], req.product, req.season, req.marketing, req.competitor_price)
        return -(price[0] - req.cost) * d
    result = minimize(neg_profit, x0=[(req.min_price+req.max_price)/2], bounds=[(req.min_price, req.max_price)], method="L-BFGS-B")
    op = float(round(result.x[0], 2))
    od = float(round(predict_demand(op, req.product, req.season, req.marketing, req.competitor_price), 1))
    opr = float(round((op - req.cost) * od, 2))
    margin = float(round(((op - req.cost) / op) * 100, 1))
    prices = [float(round(req.min_price + (req.max_price-req.min_price)*i/20, 2)) for i in range(21)]
    demands = [float(round(predict_demand(p, req.product, req.season, req.marketing, req.competitor_price), 1)) for p in prices]
    profits = [float(round((p-req.cost)*d, 2)) for p,d in zip(prices,demands)]
    return {"optimal_price": op, "expected_demand": od, "expected_profit": opr, "profit_margin": margin, "risk_range": {"low": float(round(opr*0.85,2)), "high": float(round(opr*1.15,2))}, "curve": {"prices": prices, "demands": demands, "profits": profits}}
