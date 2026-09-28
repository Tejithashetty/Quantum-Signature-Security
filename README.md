# Quantum-Inspired Cyber Threat Detection for Digital Signature Security

## Overview

This project presents a hybrid cybersecurity framework for detecting suspicious activities and cyber threats associated with digital signature verification.

The system combines statistical anomaly detection, machine learning, quantum-inspired feature analysis, adaptive thresholding, and automated security alerts into a unified detection pipeline.

A Streamlit dashboard is provided for interactive analysis and visualization.

## Problem Statement

Traditional digital signature verification primarily determines whether a signature is valid or invalid. However, repeated verification failures, replay behavior, unusual request bursts, and other abnormal patterns can indicate potential cyber attacks.

This project analyzes security-event behavior around digital signature operations to identify suspicious activities and generate security alerts.

## Key Features

- Digital signature generation and verification using ECDSA
- SHA-256 message digest generation
- Security-event generation and analysis
- Feature engineering with 14 detection features
- Statistical anomaly detection
- Isolation Forest machine-learning detection
- Quantum-inspired feature representation
- Qiskit-based quantum circuit simulation
- Hybrid threat scoring
- Adaptive thresholding
- Automated security alerts
- Attack-wise detection analysis
- Interactive Streamlit dashboard
- Evaluation metrics and visualizations

## Attack Scenarios

The experimental dataset contains:

| Event Type | Number |
|---|---:|
| Normal | 100 |
| Tampering | 30 |
| Replay | 30 |
| Verification Burst | 20 |
| **Total** | **180** |

## Detection Framework

The system extracts behavioral and signature-related features from security events.

The 14 features are:

1. message_size
2. signature_size
3. message_entropy
4. verification_failure
5. replay_intensity
6. signature_message_ratio
7. frequency_score
8. digest_entropy
9. verification_attempt_rate
10. burst_indicator
11. replay_indicator
12. signature_validity_signal
13. verification_failure_rate
14. message_size_deviation

The detection pipeline consists of:

```text
Security Events
       |
       v
Feature Engineering
       |
       +------------------+
       |                  |
       v                  v
Statistical Detector   ML Detector
       |                  |
       +--------+---------+
                |
                v
      Quantum-Inspired
       Feature Analysis
                |
                v
         Hybrid Scoring
                |
                v
       Adaptive Threshold
                |
                v
       Security Alert