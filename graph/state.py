from typing import TypedDict, Optional, Dict, List, Any, Annotated
from operator import add

class HCCDiagnosisState(TypedDict, total=False):

    patient_id: str
    skip_imaging: bool
    skip_proteomics: bool
    use_rag: bool
    is_test: bool

    mri_paths: Optional[List[str]]
    clinical_data: Optional[Dict]
    proteomics_data: Optional[Dict]
    clinical_summary: Optional[str]
    proteomics_summary: Optional[str]
    has_mri: bool
    has_clinical: bool
    has_proteomics: bool

    imaging_report: Optional[str]
    imaging_report_dict: Optional[Dict]
    imaging_cot: Optional[str]
    imaging_status: str

    clinical_report: Optional[str]
    clinical_report_dict: Optional[Dict]
    clinical_cot: Optional[str]
    clinical_status: str

    proteomics_report: Optional[str]
    proteomics_report_dict: Optional[Dict]
    proteomics_encoding: Optional[Dict]
    proteomics_cot: Optional[List[Dict]]
    proteomics_status: str

    attending_report: Optional[Dict]
    attending_cot: Optional[str]
    attending_status: str

    recurrence_prediction: Optional[Dict]
    recurrence_status: str

    recurrence_time_prediction: Optional[Dict]
    recurrence_time_status: str

    expert_opinions: Optional[Dict]

    inference_result: Optional[Dict]
    inference_status: str

    evidence_report: Optional[Dict]
    evidence_text: Optional[str]

    similar_patients: Optional[List[Any]]

    errors: Annotated[List[str], add]
