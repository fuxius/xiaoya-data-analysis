# AICare Clinical Decision Support System - CHI Paper Data Analysis

## Project Overview

This repository contains the data analysis code and documentation for the CHI paper:

**"Augmenting Clinical Decision-Making with an Interactive and Interpretable AI Copilot: A Real-World User Study with Physicians in Nephrology and Obstetrics"**

## Study Background

We conducted a comprehensive user study of the AICare system across 3 departments in 2 hospitals:

1. **Peking University Third Hospital - Obstetrics Department**: Predicting spontaneous preterm birth in pregnant women
2. **Peking University Third Hospital - Nephrology Department**: Predicting 1-year mortality risk in ESRD patients
3. **Xuzhou First People's Hospital - Nephrology Department**: Predicting 1-year mortality risk in ESRD patients

## AICare System Features

The AICare system provides 4 main interpretable AI features to assist clinical decision-making:

1. **Dynamic Risk Trajectory Visualization**: Line charts showing risk predictions across patient visits
2. **Interactive Individualized Key Indicator Analysis**: Feature importance analysis for each visit with interactive exploration
3. **Population-level Indicator Analysis Visualization**: 3D and 2D visualizations of feature importance, values, and patient risks across the dataset
4. **LLM-driven Clinical Recommendations**: Analysis and recommendations using DeepSeek-V3.1 based on patient EHR data, risk trajectories, and clinical indicators

## Study Design

### Methodology
- **Within-Subjects Design**: Each participant experienced both conditions (with and without AI assistance)
- **Counterbalancing**: Participants randomly assigned to different order groups to eliminate order effects
- **Sample Size**: 10 patient cases per department (balanced positive/negative samples from test set)

### Data Collection
1. **Pre-study**: System demonstration and background questionnaire
2. **Clinical Analysis**: 
   - Condition A (Control): Analysis without AI assistance
   - Condition B (Experimental): Analysis with AICare system
   - NASA-TLX workload assessment after each condition
3. **Post-study**: System Usability Scale (SUS), Automation Trust Scale, AICare functionality feedback
4. **Optional**: Semi-structured interviews (especially with senior physicians)

## Repository Structure

```
xiaoya-data-analysis/
├── data/                          # Raw data files (git-ignored for privacy)
│   ├── questionnaire_collection.xlsx  # Manual records of questionnaire administration
│   └── questionnaire_responses.xlsx   # Collected survey responses
├── scripts/                       # Data processing and analysis scripts
├── results/                       # Analysis outputs and visualizations
├── docs/                         # Additional documentation
└── README.md                     # This file
```

## Data Files

### Raw Data
- **questionnaire_collection.xlsx**: Contains metadata about questionnaire administration (completion times, interview notes, etc.)
- **questionnaire_responses.xlsx**: Contains the actual survey responses from participants

### Processed Data
- **merged_dataset.xlsx**: Combined dataset created by the merge script
- Additional processed files will be generated during analysis

## Getting Started

### Prerequisites
```bash
pip install pandas openpyxl numpy matplotlib seaborn scipy
```

### Data Processing
1. Run the merge script to combine raw data files:
```bash
python scripts/merge_questionnaire_data.py
```

2. Execute analysis scripts (to be developed):
```bash
python scripts/analyze_nasa_tlx.py
python scripts/analyze_sus_scores.py
python scripts/analyze_trust_metrics.py
```

## Analysis Plan

### Quantitative Analysis
- NASA-TLX workload comparison (with vs without AI)
- System Usability Scale (SUS) scores
- Automation Trust Scale metrics
- Performance metrics (accuracy, time, confidence)
- Statistical significance testing

### Qualitative Analysis
- Thematic analysis of semi-structured interviews
- Feature usage patterns
- User feedback categorization

## Data Privacy and Ethics

- All patient data has been de-identified
- The `data/` folder is git-ignored to protect sensitive information
- Study approved by institutional review boards of participating hospitals

## Contributors

- Research team from the AICare project
- Physicians from Peking University Third Hospital and Xuzhou First People's Hospital

## Citation

If you use this code or reference this study, please cite:

```bibtex
@inproceedings{aicare2025chi,
  title={Augmenting Clinical Decision-Making with an Interactive and Interpretable AI Copilot: A Real-World User Study with Physicians in Nephrology and Obstetrics},
  author={[Authors to be added]},
  booktitle={Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems},
  year={2025}
}
```

## License

This project is licensed under [License to be determined] - see the LICENSE file for details.

## Contact

For questions about this research or data analysis, please contact [Contact information to be added].