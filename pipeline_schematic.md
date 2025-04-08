```mermaid
flowchart TD
    %% Input files
    SIRV[SIRV Reads\n(FASTQ/BAM)]:::input
    SC[scRNA-seq Reads\n(FASTQ)]:::input
    
    %% Processing steps - left branch
    SIRV --> MAP[Map to SIRV\nReference]:::process
    MAP --> TRANS[Transcript\nAssignment]:::process
    
    %% Processing steps - right branch
    SC --> CELL[Extract Cell\nBarcodes & UMIs]:::process
    CELL --> COV[Model Coverage\nBias]:::process
    
    %% Integration
    TRANS --> INT[Integrate SIRV Reads with\nCell Barcodes & UMIs]:::process
    COV --> INT
    
    %% Output
    INT --> COMB[Combined FASTQ with\nGround Truth Tracking]:::output
    
    %% Evaluation (optional)
    COMB --> EVAL[Evaluate with FLAMES\n(Optional)]:::process
    EVAL --> REPORT[Generate Evaluation\nReports & Visualizations]:::output
    
    %% Define styles
    classDef input fill:#8DD3C7,stroke:#333,stroke-width:2px
    classDef process fill:#BEBADA,stroke:#333,stroke-width:2px
    classDef output fill:#FB8072,stroke:#333,stroke-width:2px
    
    %% Add legend
    LEGEND_IN[Input Files]:::input
    LEGEND_PROC[Processing Steps]:::process
    LEGEND_OUT[Output Files]:::output
    
    %% Position legend at bottom
    subgraph LEGEND[Legend]
        LEGEND_IN
        LEGEND_PROC
        LEGEND_OUT
    end
``` 