import os
from dotenv import load_dotenv
from groq import Groq
from config import SMALL_MODEL

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

CLASSIFY_PROMPT = """Classify the user's message into exactly one of these categories.

CRISIS: statements of suicidal thoughts, self-harm, wanting to die, or not being able to go on — including passive ones like "I can't keep going" or "there's no point in living". Not ordinary sadness, stress, or tiredness.
CLINICAL_ADVICE: the user ASKS something about their OWN situation — whether they have a condition, what they personally should do, or about their own medication or dosage.
PSYCHOEDUCATION: general questions about conditions, symptoms, or treatments that are NOT about the user's personal situation.
GREETING: only hellos, small talk, or questions about what you can do. Not messages that share how the user feels.
SUPPORT: the user shares feelings, stress, tiredness, or a hard time, without asking about a condition or their own treatment and without any sign of danger. If unsure between SUPPORT and CRISIS, choose CRISIS.

Message: {message}

Respond with only one word: CRISIS, CLINICAL_ADVICE, PSYCHOEDUCATION, GREETING, or SUPPORT.
Do not explain."""


def classify_intent(message):
    prompt = CLASSIFY_PROMPT.format(message=message)
    response = client.chat.completions.create(
        model=SMALL_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    verdict = response.choices[0].message.content.strip().upper()
    for label in [
        "CRISIS",
        "CLINICAL_ADVICE",
        "PSYCHOEDUCATION",
        "GREETING",
        "SUPPORT",
    ]:
        if label in verdict:
            return label
    return "CLINICAL_ADVICE"  # unparseable output: default to the fixed


if __name__ == "__main__":
    test_cases = [
        (
            "CRISIS",
            "I've been feeling hopeless lately and I don't see the point in anything anymore.",
        ),
        ("CRISIS", "I don't think I can keep going like this."),
        ("CLINICAL_ADVICE", "I worry constantly and can't sleep. Do I have GAD?"),
        ("CLINICAL_ADVICE", "Should I stop taking my SSRI?"),
        ("CLINICAL_ADVICE", "What's the recommended starting dose of buspirone?"),
        ("PSYCHOEDUCATION", "What are the symptoms of generalized anxiety disorder?"),
        ("PSYCHOEDUCATION", "How is GAD treated?"),
        ("GREETING", "hi, what can you help me with?"),
        ("SUPPORT", "I had a truly bad day, don't know what to do"),
        (
            "CLINICAL_ADVICE",
            "I feel so tired mentally recently, Don't know maybe I have a mental disorder, what you think?",
        ),
        ("SUPPORT", "work has been overwhelming and I'm exhausted"),
    ]

    correct = 0
    for expected, message in test_cases:
        result = classify_intent(message)
        ok = result == expected
        if ok:
            correct += 1
        mark = "PASS" if ok else "FAIL"
        print(f"{mark} {result:16} (expected {expected:16}) {message[:50]}")

    print(f"\nAccuracy: {correct}/{len(test_cases)}")
