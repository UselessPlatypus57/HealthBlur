"""Attribute rules derived from DICOM PS3.15 Annex E (Basic Application
Level Confidentiality Profile), Table E.1-1.

Actions:
  X  remove the attribute
  Z  replace with a zero-length value (attribute kept, type 2)
  D  replace with a non-identifying dummy / pseudonym
  U  replace UID with a consistent new UID
  C  clean: keep if no PHI is detected in the value, otherwise remove
  K  keep
  DATE  remove-or-empty under the Basic Profile; shifted / kept per date mode
  CHAR  patient characteristic: kept with ``retain_patient_characteristics``, else X

This is a curated subset covering the attributes seen in practice, backed by
catch-all rules (all PN values, all private tags, overlays, curves, unknown
free text is cleaned). It is not a certified implementation of the profile.
"""

from __future__ import annotations

# keyword -> action
RULES: dict[str, str] = {
    # Patient identity
    "PatientName": "D",
    "PatientID": "D",
    "IssuerOfPatientID": "X",
    "PatientBirthDate": "DATE",
    "PatientBirthTime": "X",
    "PatientSex": "CHAR",
    "PatientAge": "CHAR",
    "PatientSize": "CHAR",
    "PatientWeight": "CHAR",
    "PatientSexNeutered": "CHAR",
    "SmokingStatus": "CHAR",
    "PregnancyStatus": "CHAR",
    "OtherPatientIDs": "X",
    "OtherPatientNames": "X",
    "OtherPatientIDsSequence": "X",
    "PatientBirthName": "X",
    "PatientAddress": "X",
    "PatientMotherBirthName": "X",
    "PatientTelephoneNumbers": "X",
    "PatientTelecomInformation": "X",
    "EthnicGroup": "X",
    "Occupation": "X",
    "AdditionalPatientHistory": "X",
    "PatientComments": "X",
    "MedicalRecordLocator": "X",
    "PatientInsurancePlanCodeSequence": "X",
    "PatientPrimaryLanguageCodeSequence": "X",
    "MilitaryRank": "X",
    "BranchOfService": "X",
    "CountryOfResidence": "X",
    "RegionOfResidence": "X",
    "PatientReligiousPreference": "X",
    "MedicalAlerts": "X",
    "Allergies": "X",
    "ReferencedPatientPhotoSequence": "X",
    "ResponsiblePerson": "X",
    "ResponsibleOrganization": "X",
    "PatientState": "X",
    "AdmissionID": "X",
    "CurrentPatientLocation": "X",
    "PatientInstitutionResidence": "X",
    "VisitComments": "X",
    "ServiceEpisodeID": "X",
    "ServiceEpisodeDescription": "X",
    "AdmittingDiagnosesDescription": "X",
    "AdmittingDiagnosesCodeSequence": "X",
    # Study / visit identifiers
    "AccessionNumber": "Z",
    "StudyID": "Z",
    "IssuerOfAccessionNumberSequence": "X",
    "RequestAttributesSequence": "X",
    "RequestedProcedureID": "X",
    "ScheduledProcedureStepID": "X",
    "PerformedProcedureStepID": "X",
    "PlacerOrderNumberImagingServiceRequest": "X",
    "FillerOrderNumberImagingServiceRequest": "X",
    "ReferencedStudySequence": "X",
    "ReferencedPerformedProcedureStepSequence": "X",
    "ReferencedPatientSequence": "X",
    "OriginalAttributesSequence": "X",
    "ContentSequence": "X",
    "AcquisitionContextSequence": "X",
    # People
    "ReferringPhysicianName": "Z",
    "ReferringPhysicianAddress": "X",
    "ReferringPhysicianTelephoneNumbers": "X",
    "ReferringPhysicianIdentificationSequence": "X",
    "PhysiciansOfRecord": "X",
    "PhysiciansOfRecordIdentificationSequence": "X",
    "PerformingPhysicianName": "X",
    "PerformingPhysicianIdentificationSequence": "X",
    "NameOfPhysiciansReadingStudy": "X",
    "PhysiciansReadingStudyIdentificationSequence": "X",
    "OperatorsName": "X",
    "OperatorIdentificationSequence": "X",
    "RequestingPhysician": "X",
    "RequestingService": "X",
    "ScheduledPerformingPhysicianName": "X",
    "NamesOfIntendedRecipientsOfResults": "X",
    "PersonName": "X",
    "PersonAddress": "X",
    "PersonTelephoneNumbers": "X",
    # Institution and device
    "InstitutionName": "X",
    "InstitutionAddress": "X",
    "InstitutionCodeSequence": "X",
    "InstitutionalDepartmentName": "X",
    "StationName": "X",
    "DeviceSerialNumber": "X",
    "PlateID": "X",
    "GeneratorID": "X",
    "CassetteID": "X",
    "GantryID": "X",
    "DetectorID": "X",
    "SourceSerialNumber": "X",
    # Free text: removed (comments) or cleaned (descriptors)
    "ImageComments": "X",
    "IdentifyingComments": "X",
    "StudyComments": "X",
    "AcquisitionComments": "X",
    "DerivationDescription": "C",
    "InterpretationText": "X",
    "TextString": "X",
    "TextValue": "X",
    "StudyDescription": "C",
    "SeriesDescription": "C",
    "ProtocolName": "C",
    "PerformedProcedureStepDescription": "C",
    "RequestedProcedureDescription": "C",
    "ScheduledProcedureStepDescription": "C",
    "BodyPartExamined": "C",
    "ReasonForStudy": "X",
    "ReasonForTheRequestedProcedure": "X",
    # Dates and times
    "StudyDate": "DATE",
    "SeriesDate": "DATE",
    "AcquisitionDate": "DATE",
    "ContentDate": "DATE",
    "InstanceCreationDate": "DATE",
    "AcquisitionDateTime": "DATE",
    "OverlayDate": "DATE",
    "CurveDate": "DATE",
    "PerformedProcedureStepStartDate": "DATE",
    "PerformedProcedureStepEndDate": "DATE",
    "DateOfSecondaryCapture": "DATE",
    "DateOfLastCalibration": "DATE",
    "StudyTime": "TIME",
    "SeriesTime": "TIME",
    "AcquisitionTime": "TIME",
    "ContentTime": "TIME",
    "InstanceCreationTime": "TIME",
    "OverlayTime": "TIME",
    "CurveTime": "TIME",
    "PerformedProcedureStepStartTime": "TIME",
    "PerformedProcedureStepEndTime": "TIME",
    "TimeOfSecondaryCapture": "TIME",
    "TimeOfLastCalibration": "TIME",
}

# Under the Basic Profile, type-2 attributes must stay present but empty.
TYPE2_EMPTY = {"PatientName", "PatientID", "PatientBirthDate", "PatientSex", "StudyDate", "StudyTime",
               "AccessionNumber", "StudyID", "ReferringPhysicianName", "ContentDate", "ContentTime"}

# UIDs that identify standards (SOP classes, transfer syntaxes) and must be kept.
KEEP_UID_KEYWORDS = {
    "SOPClassUID", "MediaStorageSOPClassUID", "TransferSyntaxUID", "ImplementationClassUID",
    "ReferencedSOPClassUID", "ReferencedSOPClassUIDInFile", "CodingSchemeUID",
    "RelatedGeneralSOPClassUID", "OriginalSpecializedSOPClassUID", "ContextGroupExtensionCreatorUID",
    "SOPClassesInStudy", "RetrieveLocationUID",
}

# Free-text VRs that are scanned (cleaned) when no explicit rule applies.
TEXT_VRS = {"LO", "SH", "ST", "LT", "UT", "UC", "AE"}

# Attributes the evaluator expects to be preserved (clinical utility).
CLINICAL_KEEP = (
    "Modality", "Rows", "Columns", "BitsAllocated", "BitsStored", "PhotometricInterpretation",
    "PixelSpacing", "SliceThickness", "KVP", "BodyPartExamined", "SeriesDescription",
    "Manufacturer", "WindowCenter", "WindowWidth", "RescaleSlope", "RescaleIntercept",
    "ImagePositionPatient", "ImageOrientationPatient", "PatientSex", "PatientAge",
)
