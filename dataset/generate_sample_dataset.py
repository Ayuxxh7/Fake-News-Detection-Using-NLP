"""
generate_sample_dataset.py
--------------------------
Creates a SMALL SAMPLE dataset (dataset/news.csv) so the project can run
immediately without downloading anything.

IMPORTANT (be honest in your viva / report):
    This is a synthetic, template-based sample dataset (250 REAL-style and
    250 FAKE-style items). REAL-style items are written in a neutral,
    reporting tone; FAKE-style items are written in a sensational,
    clickbait / conspiracy tone. It is NOT real-world news data, so the
    scores measured on it will look very high and do not show how the model
    would perform on real news. For real results, replace news.csv with a
    real dataset (see README.md).

Run:  python dataset/generate_sample_dataset.py
"""

import csv
import random
from pathlib import Path

random.seed(42)
N_PER_CLASS = 250

REAL_OPENERS = [
    "The central bank said on Tuesday that it will keep interest rates unchanged for the next quarter",
    "The health ministry announced that a new vaccination drive will begin in district hospitals next month",
    "Researchers at the state university published a study showing that daily walking can reduce the risk of heart disease",
    "City officials confirmed that the new metro line will open to passengers after final safety inspections",
    "The finance ministry reported that tax collection rose by eight percent compared with the previous year",
    "The election commission released the schedule for the upcoming local body elections",
    "A parliamentary committee reviewed the draft education bill and asked for feedback from teachers associations",
    "The space agency successfully tested the engine of its next generation launch vehicle",
    "Local police arrested two suspects in connection with a series of bicycle thefts in the neighbourhood",
    "The transport department introduced new rules requiring all commercial vehicles to install GPS trackers",
    "The national statistics office said that retail inflation eased slightly in the last month",
    "Engineers completed repairs on the damaged bridge, and traffic has resumed in both directions",
    "The weather department forecast heavy rainfall in coastal areas over the weekend",
    "A team of doctors at the regional hospital performed a successful kidney transplant on a teenage patient",
    "The sports federation named the squad for the upcoming international cricket series",
    "The city council approved a budget for repairing roads and improving drainage systems",
    "Scientists reported a new species of frog in the forests of the Western Ghats",
    "The railway board announced additional trains for the festival season to reduce crowding",
    "The agriculture ministry increased the minimum support price for wheat for the coming season",
    "A technology company unveiled a new smartphone and said sales will start in the first week of next month",
    "The university released the results of the entrance examination on its official website",
    "The water board said supply will be interrupted in several areas on Thursday for pipeline maintenance",
    "International observers praised the orderly conduct of voting in the regional election",
    "The stock market closed slightly higher on Monday, led by gains in banking and technology shares",
    "The education department opened applications for scholarships for students from low income families",
]

REAL_DETAILS = [
    "Officials said the decision followed several weeks of consultation with experts and stakeholders.",
    "According to the official statement, the plan will be reviewed again after six months.",
    "The report, which covers data from the past three years, was reviewed by independent analysts.",
    "A spokesperson told reporters that more details will be shared at a press conference on Friday.",
    "The announcement was made during a scheduled briefing attended by journalists from several news organisations.",
    "Experts noted that the long term effects will become clear only after further data is collected.",
    "The project is expected to cost about two billion rupees and is funded through the annual budget.",
    "Opposition members asked for more transparency, while the government said all documents are public.",
    "Residents were advised to check the official website for updates and the latest guidelines.",
    "Data from a survey of about two thousand households was used in the analysis.",
    "Officials said the numbers are provisional and may be revised when the final figures are released.",
    "The agency said it would publish a detailed report once the review is complete.",
    "Analysts said the move was in line with expectations and is unlikely to cause major changes in the market.",
    "Several experts welcomed the step but cautioned that implementation will be important.",
    "The statement was published on the official website and confirmed by two senior officials.",
]

REAL_CLOSINGS = [
    "The next review meeting is scheduled for later this month.",
    "Further updates are expected after the committee submits its findings.",
    "Citizens can send their suggestions through the official portal until the end of the month.",
    "The department said it will continue to monitor the situation closely.",
    "Reporters were told that the proceedings were recorded and the minutes will be made public.",
    "The authorities did not comment on the matter beyond the official statement.",
    "The figures will be updated on the website every week.",
    "More information is available from the district office during working hours.",
    "The announcement comes after similar steps taken in neighbouring states last year.",
    "Work is expected to be completed within the stated timeline, officials added.",
]

FAKE_OPENERS = [
    "SHOCKING!!! Government secretly puts tracking chips in the new vaccine and doctors are being forced to stay silent",
    "You will not believe what this famous celebrity did, and the media is desperately trying to hide it from you",
    "BREAKING: Scientists confirm that drinking hot lemon water cures every disease in just three days",
    "Aliens landed near the capital last night and the army is hiding the proof from the public",
    "Mind blowing secret trick lets you earn lakhs of rupees from home with zero effort, banks hate it",
    "EXPOSED: The election was completely rigged and millions of fake votes were counted, insiders reveal",
    "Miracle pill melts belly fat overnight and doctors do not want you to know about it",
    "Famous politician caught in giant secret scandal, leaked video proves everything, mainstream media silent",
    "URGENT WARNING: Eating this common food will kill you within hours, share with everyone you love",
    "Secret group of billionaires is planning to control the weather and destroy your crops, whistleblower claims",
    "Government will cancel all currency notes tomorrow night, a forwarded message from a top insider says",
    "This one weird home remedy makes cancer disappear and big pharma is furious",
    "Shocking truth: the moon landing was filmed in a studio and a former employee finally confesses",
    "Free smartphones for everyone! Click this link now and claim your gift before the offer ends tonight",
    "5G towers are spreading a deadly virus and the authorities are lying to you",
    "Cricket star secretly dies in accident but the board is hiding the news to protect sponsors",
    "Scientists are stunned: man lives for 200 years by drinking only one special tea",
    "Hidden camera reveals what the government is really putting in your drinking water",
    "Unbelievable! Child genius invents machine that gives free electricity forever, powerful companies try to stop him",
    "Doctors shocked as simple kitchen spice reverses aging by twenty years in one week",
    "Leaked documents prove the whole economy will crash tomorrow and banks will freeze all your money",
    "WHO is hiding the real cure and the truth will make you furious",
    "Anonymous source says famous actor is secretly an alien spy sent to control our minds",
    "Thousands of people are quietly disappearing and the news channels are told to say nothing",
    "Share this message to ten people in the next hour or terrible bad luck will follow you forever",
]

FAKE_DETAILS = [
    "Everyone is talking about this and you must read until the end to learn the real truth.",
    "A friend of a friend who works inside the department told us everything, so it must be true.",
    "The shocking proof was deleted from the internet within minutes, but we saved a copy.",
    "No official will ever admit this, but thousands of people have already seen it with their own eyes.",
    "Experts do not want you to know this simple secret that changes everything.",
    "This is the biggest cover up in history and the mainstream media will never report it.",
    "Anonymous insiders say that the powerful elite are behind everything, but no one dares to say it aloud.",
    "Click here to see the unbelievable video before it is banned everywhere.",
    "Hundreds of comments on social media confirm that the story is one hundred percent real.",
    "They are lying to us every day, and it is time that the people wake up.",
    "Natural cure discovered by a village grandmother beats all modern medicine, no tests needed.",
    "A viral post claims the miracle result was proven, though no source or study is given.",
    "The truth is finally out, and it is worse than anyone could ever imagine.",
    "Do not trust the official story because it is all a big lie made up by corrupt leaders.",
    "You will be amazed, scared and angry after reading what happens next.",
]

FAKE_CLOSINGS = [
    "Share this with everyone before it gets deleted!!!",
    "Forward to all your groups right now, every second counts.",
    "Wake up people, the real news is only here and nowhere else.",
    "Like and share if you care about your family, ignore if you want to be the next victim.",
    "Remember, you saw it here first, and they cannot stop us from spreading the truth.",
    "This message is going viral, so copy and paste it everywhere today.",
    "Tell your friends and family immediately, the government will try to delete this page.",
    "Do not wait, act now, and spread the word to save lives.",
    "Only real patriots will share this post.",
    "Thousands have already shared it, so be part of the movement before it is too late.",
]


def build_articles(openers, details, closings, n):
    """Build n UNIQUE articles by combining an opener, a detail and a closing."""
    articles = set()
    while len(articles) < n:
        parts = [random.choice(openers)]
        if random.random() < 0.85:       # sometimes skip the detail (short article)
            parts.append(random.choice(details))
        if random.random() < 0.70:       # sometimes skip the closing
            parts.append(random.choice(closings))
        text = ". ".join(p.rstrip(".") for p in parts)
        if not text.endswith(("!", "?")):
            text += "."
        articles.add(text)
    return sorted(articles)


def main():
    rows = []
    for text in build_articles(REAL_OPENERS, REAL_DETAILS, REAL_CLOSINGS, N_PER_CLASS):
        rows.append((text, "REAL"))
    for text in build_articles(FAKE_OPENERS, FAKE_DETAILS, FAKE_CLOSINGS, N_PER_CLASS):
        rows.append((text, "FAKE"))
    random.shuffle(rows)

    out_path = Path(__file__).resolve().parent / "news.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
