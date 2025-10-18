# COHESION

Code and dataset for the paper "Explaining Cohesiveness of Social Communities: A Group Cohesion Theory-based Approach"


## Folder Structure
    .
    ├── COHESION           # Scripts for three COHESION frameworks (IO-COHESION, I-COHESION_w, and I-COHESION)
    ├── Competitors           # Scripts for generating competitors' explanations
    ├── Datasets           # Datasets used for calculation
    ├── Experiments        # Some scripts for conducting experiments
    ├── Figures            # Figures
    └── README.md


### Competitors
--> Feature.py: A feature-based method providing post-hoc explanations for communities based on interpretable network features, identifying explanatory features at both the node and node-pair levels. Refer to its [codebase](https://github.com/sophiefsadler/community_finding/tree/master), graph feature computation component was extracted.
--> Prototype.py: A prototype-based method explaining communities using a central node and its radius. We adapted its [original codebase](https://github.com/xuannnn523/CCTS) with modified inputs. Since its explanation consists only of a central node and an influence radius, other unrelated output information was commented out.

### Datasets
The online social network datasets and communities that we consider in our study, can be found [here](https://github.com/Yningg/Cohesion_Evaluation/tree/main/Algorithm_Output).

### Experiments
Integrate_Results.py: Extract and integrate communities identified by different algorithms for each dataset. Extracted and integrated communities are stored under the same folder *./Datasets/Communities*.