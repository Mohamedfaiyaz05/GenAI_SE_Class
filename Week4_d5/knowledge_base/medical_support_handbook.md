# MediCare Plus — Patient Support Knowledge Handbook

**Version:** 1.0  
**Type:** Fictional training knowledge base for a CrewAI customer-support demo  
**Important:** MediCare Plus is fictional. The administrative policies below are invented for software testing, not real provider policies. General medical information is educational and does not replace instructions from a licensed clinician.

## 1. Scope of the support assistant
The assistant can explain MediCare Plus demo appointment policies, test-preparation basics, report access, and how to contact a care team. It must not diagnose conditions, interpret an individual's test results, prescribe drugs, or recommend changing prescribed medicines. For severe or potentially life-threatening symptoms, advise immediate local emergency medical assistance.

## 2. Appointment booking (fictional clinic policy)
Appointments may be requested through the clinic's patient portal or by calling its reception desk. The demo clinic does not provide a real booking URL or phone number. Appointment requests are confirmed only after the clinic sends a confirmation. Walk-in availability is not guaranteed.

## 3. Rescheduling and cancellations (fictional clinic policy)
Patients can request rescheduling or cancellation at least 24 hours before a scheduled appointment without a demo cancellation fee. Requests made within 24 hours may require staff review. The assistant must not claim that a booking was changed unless a connected scheduling system confirms it.

## 4. Blood-test fasting
Not all blood tests require fasting. Whether fasting is needed, and for how long, depends on the specific test and the ordering clinician's instructions. When fasting is required, plain water is generally allowed unless the care team specifically instructs otherwise. Do not eat, drink beverages other than permitted water, or alter fasting instructions without checking with the care team. If a patient accidentally eats during a required fast, they should inform the laboratory before the test.

## 5. Medicines before laboratory tests
Patients should not stop or change prescribed medications on their own before a blood test. They should ask the ordering clinician or laboratory whether any medication or supplement affects test preparation. The assistant must not provide individualized medication instructions.

## 6. Laboratory report access (fictional clinic policy)
The demo clinic makes completed reports available in its patient portal after clinical release. Some results may take longer because different tests have different processing times. The assistant must not promise an exact turnaround time unless an authorized clinic system provides one. For missing reports, patients should contact the clinic using a verified channel.

## 7. Report interpretation
Reference ranges and result meanings vary by test, laboratory, medical history, and clinical context. The assistant may explain general terms but must not tell a patient that their own result confirms or rules out a diagnosis. Patients should discuss personal results with their clinician.

## 8. Preparing for a visit (fictional clinic policy)
Patients should bring their appointment confirmation, available referral or test order, a list of current medications and supplements, and identification if required by the clinic. The assistant must not request identity documents or sensitive medical records in public chat.

## 9. Privacy and records
The assistant must not disclose patient-specific records without authenticated, authorized access to the relevant clinical system. A public web search is never an acceptable substitute for patient-record access. Avoid including personal health information in analytics or audit logs.

## 10. Emergency escalation
If a person reports severe chest pain, major difficulty breathing, signs of stroke, uncontrolled bleeding, or another possible emergency, advise them to seek immediate emergency medical assistance. Do not delay emergency advice to run vector search or web search.

## 11. Out-of-scope knowledge and fallback
General educational questions not answered by this handbook may be researched through reputable public medical sources, such as MedlinePlus, NIH, NHS, or relevant government health authorities. External findings must include a source URL and must be presented as general information, not personalized medical advice. Questions about MediCare Plus's own policies cannot be answered from unrelated public websites.

## 12. Refunds and payment reversals (fictional clinic policy)
For this demonstration clinic, a patient who cancels a prepaid appointment at least 24 hours before its scheduled start may request a full refund of the appointment fee. Cancellations within 24 hours are reviewed by clinic staff and are not automatically eligible for a refund. If the clinic cancels an appointment and cannot offer an acceptable replacement, the patient may request a full refund. Completed consultations and tests are generally non-refundable; billing errors or duplicate charges should be reviewed by the billing team. Approved refunds are initiated to the original payment method within 7 business days, but the payment provider may take additional time to credit the account. The assistant must never promise a refund, claim it has been approved, or request full card or banking details. Refund eligibility and status require verification through an authorized clinic billing system; SerpAPI cannot verify this clinic's refunds.

## 13. Knowledge provenance
This is a synthetic demo document. The administrative rules are fabricated. General blood-test preparation principles are consistent with MedlinePlus's educational guidance on fasting before a blood test: https://medlineplus.gov/lab-tests/fasting-for-a-blood-test/ . Review medical statements against current authoritative guidance before deploying beyond a demonstration.
