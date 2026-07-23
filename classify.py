import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

CLASSIFY_PROMPT = """ Classify users' message exactly to one of the categories.

CRISIS: expressions of hopeless, worhtless, self-harm, suicide or that life has no point.
CLINICAL_ADVICE: asking about their OWN situation — whether they have a condition, what they personally should do, or advice about their own medication or dosages.
PSYCHOEDUCATION: general questions about conditions, symptoms, or treatments that are NOT about the user's personal situation.
GREETING: greeting, small talk or questions what you can do.

Message:{message}

respond only with one word: CRISIS, CLINICAL ADVICE, PSYCHOEDUATION OR GREETING.

Do not explain"""


def classify_intent(message):
    prompt = CLASSIFY_PROMPT.format(message=message)
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    verdict = response.choices[0].message.content.strip().upper()
    for label in ["CRISIS", "CLINICAL_ADVICE", "PSYCHOEDUCATION", "GREETING"]:
        if label in verdict:
            return label
    return "PSYCHOEDUCATION"  # safe fallback (for now)


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
