# 📊 Bond & Treasury Analysis Agent

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-yellowgreen)](https://pandas.pydata.org/)
[![BYMA API](https://img.shields.io/badge/BYMA-API-orange)](https://www.byma.com.ar/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

### Intelligent agent for analyzing Argentine sovereign bonds and treasury bills using BYMA API data.

---

## 🧩 Overview

**Bond & Treasury Analysis Agent** is an automated framework that extracts, cleans, and analyzes financial instruments listed in **BYMA (Bolsas y Mercados Argentinos)**.  
It combines structured data processing with **GPT-powered text parsing** to interpret technical sheets (*fichas técnicas*), compute **cash flow schedules**, and evaluate **yield to maturity (YTM)** and **annualized interest rates (TNA)**.

This system aims to support **data-driven investment decisions**, comparing bond returns against **repo (caución)** rates to identify optimal short- and medium-term opportunities.

---

## ⚙️ Core Features

- 🔗 **API Data Extraction**  
  Connects to BYMA’s REST API to download technical sheets and market prices.

- 🧠 **Natural Language Parsing (LLM)**  
  Uses GPT models to interpret the `interes` and `formaAmortizacion` fields, extracting:
  - Amortization schedule (bullet / installments)
  - Step-up or fixed interest rates
  - Payment frequency and key dates

- 💰 **Cash Flow Generation**  
  Automatically constructs payment schedules for each instrument, handling:
  - Step-up interest periods  
  - Monthly capitalization (e.g., TEM, TAMAR)  
  - Bullet or amortizing structures  

- 📉 **Yield Computation**  
  Produces detailed flow tables for each bond and bill, ready for:
  - Yield to Maturity (YTM) calculation  
  - Annualized rate (TNA) estimation  

- 🧮 **Market Integration**  
  Combines computed flows with **real-time prices** from the BYMA API for continuous valuation.

- 📈 **Investment Optimization**  
  Compares yields between government bonds and repo rates to optimize capital allocation.

---

## 🧱 Architecture

##