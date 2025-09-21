# 🚚 AI-Powered Delivery ETA & Trust Score Prediction

This project demonstrates how to build a **hybrid ML pipeline** for predicting **delivery ETAs** and **trust scores** on an e-commerce platform (e.g., Meesho).  
It combines **temporal models (LSTM)**, **graph models (GNN)**, and **gradient boosting (XGBoost)** to handle different data modalities.

---

## 📌 Problem
1. **Uncertain ETAs** → Customers cancel orders if delivery is late.  
2. **Low transparency** → Buyers lack visibility into seller & courier reliability.  
3. **Real-time conditions ignored** → Weather, traffic, and courier delays affect actual delivery times.

---

## 🧠 Approach
The solution is divided into **three stages**:

### 1. **ETA Prediction (LSTM + GNN)**
- **Input**:  
  - Historical delivery logs (timestamps, routes, courier IDs, order locations).  
  - Route graphs (nodes = hubs/regions, edges = routes with distances & travel times).  
- **Process**:  
  - **LSTM** captures temporal patterns (rush hours, weekends).  
  - **GNN** encodes route graphs and propagates traffic/courier reliability signals.  
- **Output**: Predicted ETA distribution for a new order.  

### 2. **Trust Score Prediction (Regression + Classification)**
- **Input**:  
  - Seller history (on-time %, cancellations).  
  - Courier stats, product type, distance.  
  - Predicted ETA from Stage 1.  
- **Process**:  
  - **Regression** maps features → numeric ETA (hours/days).  
  - **Classification (XGBoost)** predicts trust class (High / Medium / Low).  
- **Output**:  
  - ETA (e.g., 2.3 days).  
  - Trust probability (e.g., 0.95 confidence of on-time delivery).  

### 3. **Real-time Adaptation**
- **Input**: Live courier location, GPS traffic, weather forecasts.  
- **Process**: Adjust ETA dynamically (e.g., +2 hrs if heavy rain detected).  
- **Output**: Updated ETA & reliability score visible to customers.  

---

## 🏗️ Project Structure
