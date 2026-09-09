# Customer Churn Prediction — Production MLOps System

An end-to-end production-grade machine learning system for predicting customer churn, with automated model training, experiment tracking, model registry, deployment, monitoring, canary releases, rollback, and CI/CD.

## Project Overview

This project demonstrates how a machine learning model can be taken beyond experimentation and deployed as a production ML system.

The system covers the complete lifecycle:

Data → Preprocessing → Model Training → MLflow Tracking → Quality Gates → Model Registry → Canary Evaluation → Production Promotion → FastAPI Deployment → Monitoring → Retraining Decision → Promotion / Rejection / Rollback

## Key Features

- Customer churn prediction using XGBoost
- Hyperparameter tuning with GridSearchCV
- MLflow experiment tracking
- MLflow Model Registry
- Production model aliases
- Automated model quality gates
- Candidate-vs-production comparison
- Dataset SHA-256 lineage tracking
- Training environment lineage
- FastAPI inference service
- API-key authentication
- Separate admin authentication
- Rate limiting
- Request ID tracing
- Health and readiness endpoints
- Model explanation endpoint
- Docker containerization
- GitHub Actions CI/CD
- Render deployment
- Data drift monitoring
- Prediction logging
- Performance monitoring
- Automated retraining decision
- Application-level 90/10 canary routing
- Automated canary evaluation
- Automated model promotion
- Model rollback
- Canary failure cleanup

## Architecture

```text
                    Customer Data
                         |
                         v
                Data Preprocessing
                         |
                         v
               Model Training
              XGBoost + Tuning
                         |
                         v
                      MLflow
               Tracking + Registry
                         |
                  Quality Gates
                         |
                         v
                  Candidate Model
                         |
                  Canary Evaluation
                         |
                  +------+------+
                  |             |
                  v             v
               Promote       Reject
                  |             |
                  v             v
             Production      Cleanup
                  |
                  v
             FastAPI Service
                  |
             +----+----+
             |         |
           90%        10%
             |         |
             v         v
        Production   Canary
          Model       Model
             |         |
             +----+----+
                  |
                  v
             Prediction Logs
                  |
                  v
        Monitoring & Drift Detection
                  |
                  v
          Retraining Decision