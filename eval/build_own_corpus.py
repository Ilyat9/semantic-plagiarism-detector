"""Построение собственного eval-корпуса: data/eval/pairs.jsonl.

Для каждого базового текста создаются три пары:
- verbatim  — точная копия фрагмента;
- paraphrase — перефраз (часть — вручную, 3 текста — через локальную LLM Ollama);
- original  — несвязанный текст из другой темы.

Использование: python -m eval.build_own_corpus
"""

from __future__ import annotations

import json
from pathlib import Path

OUT_PATH = Path("data/eval/pairs.jsonl")
OLLAMA_MODEL = "qwen2.5:7b-instruct-q4_K_M"

# (id, базовый текст, ручной перефраз или None -> перефраз через LLM)
BASE_TEXTS: list[tuple[str, str, str | None]] = [
    (
        "photosynthesis",
        "Photosynthesis converts sunlight into chemical energy in plants. "
        "Chlorophyll absorbs light mainly in the blue and red spectra. "
        "Oxygen is released as a byproduct of splitting water molecules.",
        "Plants transform sunlight into usable chemical energy through photosynthesis. "
        "The pigment chlorophyll captures light primarily in blue and red wavelengths. "
        "As water molecules are split, oxygen is emitted as a side product.",
    ),
    (
        "french_revolution",
        "The French Revolution began in 1789 with the storming of the Bastille. "
        "It abolished the monarchy and radically reshaped French society. "
        "Its ideas of liberty and equality influenced politics across Europe.",
        "Starting with the Bastille's capture in 1789, the French Revolution overthrew "
        "the monarchy and transformed society in France profoundly. "
        "The revolutionary ideals of freedom and equality shaped European political thought.",
    ),
    (
        "photosynthesis_deep",
        "The Calvin cycle fixes carbon dioxide into organic molecules in the stroma. "
        "Rubisco catalyzes the first major step of carbon fixation. "
        "The cycle consumes ATP and NADPH produced by the light reactions.",
        "Within the stroma, the Calvin cycle incorporates carbon dioxide into organic compounds. "
        "The enzyme rubisco drives the initial step of fixing carbon. "
        "This process uses up the ATP and NADPH generated during the light-dependent reactions.",
    ),
    (
        "internet_origins",
        "The Internet originated from the ARPANET project funded by the US Department of Defense. "
        "The first message was sent between UCLA and Stanford in 1969. "
        "TCP/IP became the standard protocol suite in 1983.",
        None,  # перефраз через LLM
    ),
    (
        "black_holes",
        "A black hole is a region of spacetime where gravity is so strong that nothing can escape. "
        "The boundary of no escape is called the event horizon. "
        "Black holes form when massive stars collapse at the end of their life cycle.",
        "In a black hole, gravitational pull is so intense that not even light can get away. "
        "Its outer boundary, known as the event horizon, marks the point of no return. "
        "Such objects arise when very massive stars collapse as their lives end.",
    ),
    (
        "vaccines",
        "Vaccines train the immune system to recognize pathogens without causing disease. "
        "They typically contain weakened or inactivated parts of a microorganism. "
        "Widespread vaccination has eradicated smallpox and nearly eliminated polio.",
        "By exposing the immune system to harmless forms of a pathogen, vaccines teach it "
        "to fight the real infection. Immunization campaigns wiped out smallpox globally "
        "and brought polio to the brink of extinction.",
    ),
    (
        "machine_learning",
        "Gradient descent optimizes model parameters by following the negative gradient. "
        "The learning rate controls the size of each update step. "
        "Too large a learning rate can cause divergence instead of convergence.",
        "By moving parameters in the direction opposite to the gradient, gradient descent "
        "improves a model iteratively. Each step's magnitude is governed by the learning rate, "
        "which, if set too high, may make training diverge rather than converge.",
    ),
    (
        "honeybees",
        "Honeybees communicate the location of food sources through the waggle dance. "
        "The angle of the dance indicates direction relative to the sun. "
        "Its duration encodes the distance to the food source.",
        "To tell nestmates where food is, honeybees perform a waggle dance: its orientation "
        "shows the direction with respect to the sun, while the dance's length reflects "
        "how far away the food lies.",
    ),
    (
        "ww2_dday",
        "The Normandy landings on June 6, 1944 opened a second front in Western Europe. "
        "Allied forces crossed the English Channel in the largest amphibious invasion in history. "
        "The operation marked a turning point in the Second World War.",
        None,  # перефраз через LLM
    ),
    (
        "dna_structure",
        "DNA consists of two strands forming a double helix. "
        "The strands are held together by hydrogen bonds between complementary bases. "
        "Adenine pairs with thymine, and guanine pairs with cytosine.",
        "Two intertwined strands make up the double-helical structure of DNA. "
        "Complementary bases linked by hydrogen bonds keep the strands together: "
        "adenine binds to thymine, while guanine binds to cytosine.",
    ),
    (
        "climate_feedback",
        "Melting Arctic ice reduces the planet's albedo, causing it to absorb more sunlight. "
        "This positive feedback accelerates further warming and ice loss. "
        "Such feedback loops make climate projections highly sensitive to initial changes.",
        "As Arctic ice disappears, Earth reflects less sunlight and absorbs more of it. "
        "This self-reinforcing cycle speeds up warming and further melting, which is why "
        "climate forecasts react strongly to small initial shifts.",
    ),
    (
        "printing_press",
        "Johannes Gutenberg introduced the movable-type printing press to Europe around 1440. "
        "It dramatically reduced the cost of producing books. "
        "The resulting spread of printed material transformed education and religion.",
        None,  # перефраз через LLM
    ),
    # === Новые тексты для расширения корпуса до 35 ===
    (
        "antibiotics",
        "Antibiotics are medications that destroy or slow down bacterial growth. "
        "Penicillin, discovered by Alexander Fleming in 1928, was the first widely used "
        "antibiotic. Overuse has led to resistant strains that threaten modern medicine.",
        "These drugs kill bacteria or inhibit their reproduction. "
        "Fleming's 1928 discovery of penicillin revolutionized infection treatment. "
        "However, excessive prescription has created dangerous drug-resistant bacteria.",
    ),
    (
        "industrial_revolution",
        "The Industrial Revolution began in Britain during the late 18th century. "
        "Steam power and mechanized textile production transformed manufacturing. "
        "Urbanization accelerated as rural populations moved to factory towns.",
        None,
    ),
    (
        "quantum_computing",
        "Quantum computers use qubits that can exist in superposition of states. "
        "This allows them to perform certain calculations exponentially faster than classical "
        "computers. "
        "IBM and Google have demonstrated quantum supremacy on specific problems.",
        "Unlike classical bits, qubits leverage superposition for parallel computation. "
        "Some algorithms run exponentially faster on quantum hardware. "
        "Tech giants have achieved quantum advantage on specialized tasks.",
    ),
    (
        "roman_law",
        "Roman law forms the foundation of civil law systems used across Europe. "
        "The Twelve Tables, published around 450 BCE, codified basic rights and procedures. "
        "Justinian's Corpus Juris Civilis later systematized Roman legal principles.",
        "European civil law traces its roots to ancient Roman legal tradition. "
        "The earliest codification appeared in the Twelve Tables circa 450 BCE. "
        "Justinian later compiled comprehensive collections of Roman jurisprudence.",
    ),
    (
        "neurons",
        "Neurons are specialized cells that transmit electrical and chemical signals. "
        "Each neuron consists of a cell body, dendrites, and an axon. "
        "Synapses allow communication between neurons through neurotransmitter release.",
        "These specialized cells relay information via electrochemical impulses. "
        "A typical nerve cell contains a soma, branching dendrites, and a long axon. "
        "Chemical messengers bridge the gap between adjacent neurons at synaptic junctions.",
    ),
    (
        "inflation",
        "Inflation measures the rate at which general price levels rise over time. "
        "Central banks target low and stable inflation, typically around two percent annually. "
        "Hyperinflation can destroy savings and destabilize entire economies.",
        "Rising consumer prices over time define inflation. "
        "Monetary authorities aim to keep annual price growth near two percent. "
        "Extreme inflationary spirals can wipe out personal savings and collapse markets.",
    ),
    (
        "solar_system",
        "The solar system formed 4.6 billion years ago from a collapsing molecular cloud. "
        "Eight planets orbit the Sun, with Jupiter being the largest. "
        "The asteroid belt between Mars and Jupiter contains remnants of planetary formation.",
        None,
    ),
    (
        "shakespeare",
        "William Shakespeare wrote approximately 39 plays and 154 sonnets. "
        "His works explore universal themes of love, power, jealousy, and mortality. "
        "Hamlet's soliloquy 'To be or not to be' remains among the most quoted passages.",
        "The Bard composed nearly forty plays and over one hundred fifty sonnets. "
        "His writing delves into timeless human experiences like passion, ambition, and death. "
        "The famous monologue from Hamlet continues to be widely cited centuries later.",
    ),
    (
        "plate_tectonics",
        "Earth's lithosphere is divided into tectonic plates that float on the asthenosphere. "
        "Plate boundaries generate earthquakes, volcanoes, and mountain ranges. "
        "The theory was widely accepted after seafloor spreading was confirmed in the 1960s.",
        "The outer shell of our planet consists of moving plates riding atop softer mantle rock. "
        "Collisions and separations at plate edges create seismic activity and volcanic eruptions. "
        "Evidence of underwater ridge expansion in the mid-20th century solidified the theory.",
    ),
    (
        "cognitive_bias",
        "Confirmation bias leads people to seek information that supports existing beliefs. "
        "This psychological tendency distorts objective reasoning and decision-making. "
        "Social media algorithms often amplify confirmation bias by creating echo chambers.",
        "People naturally favor data that reinforces their preconceptions. "
        "Such selective thinking undermines rational judgment. "
        "Online platforms frequently worsen this effect by filtering content to match user views.",
    ),
    (
        "rainforest",
        "Tropical rainforests harbor more than half of Earth's terrestrial species. "
        "The Amazon basin alone contains approximately ten percent of all known species. "
        "Deforestation threatens biodiversity and contributes significantly to climate change.",
        "Over fifty percent of land-dwelling organisms live in tropical forest ecosystems. "
        "The Amazon region hosts roughly one in ten documented species worldwide. "
        "Clearing these forests endangers wildlife and releases substantial carbon emissions.",
    ),
    (
        "relativity",
        "Einstein's theory of general relativity describes gravity as curvature of spacetime. "
        "Massive objects warp the fabric of space, affecting the motion of other bodies. "
        "GPS satellites must account for relativistic effects to maintain accuracy.",
        None,
    ),
    (
        "pythagoras",
        "The Pythagorean theorem states that in a right triangle, a² + b² = c². "
        "This fundamental relation was known to ancient Babylonians but proved by Greek "
        "mathematicians. "
        "It underlies trigonometry and has countless practical applications.",
        "For right-angled triangles, the square of the hypotenuse equals the sum of squared legs. "
        "Ancient Mesopotamians knew this relationship; Greeks later provided formal proofs. "
        "Trigonometry and numerous real-world calculations depend on this principle.",
    ),
    (
        "existentialism",
        "Existentialist philosophy emphasizes individual freedom, choice, and responsibility. "
        "Sartre argued that existence precedes essence — humans define themselves through actions. "
        "Absurdism, explored by Camus, confronts the tension between human search for meaning and "
        "silent universe.",
        "This school of thought centers on personal autonomy and accountability. "
        "Sartre believed people first exist, then create their own nature through deeds. "
        "Camus examined the conflict between humanity's quest for purpose and an indifferent "
        "cosmos.",
    ),
    (
        "social_media",
        "Social media platforms use recommendation algorithms to maximize user engagement. "
        "These systems analyze behavior patterns to predict what content will capture attention. "
        "Critics argue this optimization can promote misinformation and polarization.",
        None,
    ),
    (
        "human_evolution",
        "Modern humans evolved in Africa approximately 300,000 years ago. "
        "Homo sapiens coexisted with Neanderthals and Denisovans before these cousins "
        "went extinct. Genetic analysis reveals that non-African populations carry small "
        "amounts of Neanderthal DNA.",
        "Our species emerged in Africa roughly three hundred millennia ago. "
        "Early Homo sapiens lived alongside other human species that eventually died out. "
        "DNA studies show that people outside Africa inherited traces of Neanderthal ancestry.",
    ),
    (
        "linguistics",
        "Noam Chomsky proposed that humans possess an innate universal grammar. "
        "This theory suggests children can learn any language because they are born with "
        "linguistic structures. "
        "Critics point to the vast diversity of world languages as evidence against universal "
        "patterns.",
        "Chomsky argued that language acquisition relies on built-in mental frameworks. "
        "According to this view, infants can master any tongue thanks to inherent grammatical "
        "knowledge. "
        "Opponents cite extreme variation among global languages to challenge this hypothesis.",
    ),
    (
        "pompeii",
        "The Roman city of Pompeii was buried by the eruption of Mount Vesuvius in 79 CE. "
        "Ash preserved buildings, artifacts, and bodies for nearly two millennia. "
        "Archaeological excavations since the 18th century have revealed daily life in ancient "
        "Rome.",
        "Vesuvius destroyed this Roman settlement in 79 AD with volcanic ash. "
        "The debris sealed structures, objects, and human remains for almost two thousand years. "
        "Digging since the 1700s has uncovered remarkably detailed glimpses of antiquity.",
    ),
    (
        "exoplanets",
        "Exoplanets are planets orbiting stars other than our Sun. "
        "The Kepler mission discovered thousands using the transit method. "
        "Some exoplanets lie in habitable zones where liquid water could exist on their surfaces.",
        "These worlds circle foreign stars beyond our solar system. "
        "NASA's Kepler telescope found thousands by detecting periodic dimming of starlight. "
        "Certain distant planets occupy regions where conditions might support surface oceans.",
    ),
    (
        "coral_reef",
        "Coral reefs support approximately twenty-five percent of all marine species. "
        "These ecosystems are built by colonies of tiny animals called coral polyps. "
        "Rising ocean temperatures cause coral bleaching, threatening these biodiversity hotspots.",
        "A quarter of sea creatures depend on reef ecosystems for survival. "
        "Microscopic polyp colonies construct these elaborate underwater structures over "
        "centuries. Warming seas trigger bleaching events that endanger these marine sanctuaries.",
    ),
    (
        "thunderstorms",
        "Thunderstorms form when warm, moist air rises into cooler atmospheric layers. "
        "Updrafts and downdrafts create electrical charge separation, leading to lightning. "
        "Severe storms can produce hail, tornadoes, and flash flooding.",
        "These weather systems develop as heated humid air ascends into colder upper layers. "
        "Rising and falling air masses generate static electricity that discharges as lightning "
        "bolts. "
        "Extreme convective events may spawn hailstones, twisters, and sudden inundations.",
    ),
    (
        "bats",
        "Bats are the only mammals capable of sustained flight. "
        "They use echolocation to navigate and hunt insects in complete darkness. "
        "Many bat species are pollinators and seed dispersers critical to ecosystem health.",
        None,
    ),
    (
        "ferns",
        "Ferns are vascular plants that reproduce through spores rather than seeds. "
        "They dominated Earth's forests during the Carboniferous period. "
        "Today, ferns thrive in moist, shaded environments across all continents except "
        "Antarctica.",
        "These plants lack flowers and seeds, multiplying instead through airborne spores. "
        "Three hundred million years ago, vast forests of these organisms covered the planet. "
        "Modern varieties prefer damp, dark habitats and grow on every landmass except the frozen "
        "south.",
    ),
]

# Тексты-«чужие» для original-пар (тематически не пересекаются)
UNRELATED: list[tuple[str, str]] = [
    (
        "stock_market",
        "The stock market closed higher on Tuesday amid a rally in technology shares. "
        "Investors welcomed stronger-than-expected quarterly earnings. "
        "Analysts nevertheless warned of continued volatility in commodity prices.",
    ),
    (
        "recipe",
        "To make sourdough bread, mix flour, water and an active starter. "
        "Let the dough ferment overnight at room temperature. "
        "Bake in a preheated Dutch oven for a crisp crust.",
    ),
    (
        "football",
        "The club signed a new striker for a record transfer fee. "
        "Fans filled the stadium hours before kickoff. "
        "The coach promised an attacking style of play this season.",
    ),
    (
        "gardening",
        "Tomatoes need at least six hours of direct sunlight daily. "
        "Water them deeply but infrequently to encourage strong roots. "
        "Mulch helps retain soil moisture during hot weeks.",
    ),
    (
        "travel",
        "The ancient city of Petra was carved into rose-red cliffs by the Nabataeans. "
        "Visitors enter through a narrow gorge called the Siq. "
        "The Treasury facade remains the most iconic monument.",
    ),
    (
        "music",
        "Beethoven composed his Ninth Symphony while completely deaf. "
        "The final movement features the famous Ode to Joy melody. "
        "It premiered in Vienna in 1824 to enthusiastic applause.",
    ),
    (
        "cooking",
        "Risotto requires constant stirring to release starch from Arborio rice. "
        "Hot broth is added gradually, one ladle at a time. "
        "The finished dish should be creamy yet retain a slight bite.",
    ),
    (
        "sports",
        "Marathon runners hit the wall around kilometer thirty-five. "
        "Glycogen depletion causes sudden fatigue and mental doubt. "
        "Proper pacing and nutrition help delay this phenomenon.",
    ),
    (
        "art",
        "Impressionist painters focused on capturing light and atmosphere. "
        "They used visible brushstrokes and bright, unmixed colors. "
        "Monet's water lily series exemplifies this approach.",
    ),
    (
        "fashion",
        "Coco Chanel revolutionized women's fashion by introducing trousers. "
        "She popularized the little black dress and costume jewelry. "
        "Her designs emphasized comfort and understated elegance.",
    ),
    (
        "architecture",
        "Gothic cathedrals feature pointed arches and flying buttresses. "
        "Stained glass windows tell biblical stories through colored light. "
        "Construction often spanned several generations of craftsmen.",
    ),
    (
        "chemistry",
        "The periodic table organizes elements by atomic number and properties. "
        "Mendeleev left gaps for undiscovered elements in his 1869 version. "
        "His predictions for gallium and germanium proved remarkably accurate.",
    ),
]


def llm_paraphrase(text: str) -> str:
    """Перефраз через локальную LLM (Ollama). Промпт: перепиши своими словами."""
    import httpx

    # trust_env=False: иначе httpx подхватывает системный прокси macOS,
    # и запрос к localhost уходит через прокси -> 502
    with httpx.Client(trust_env=False) as client:
        resp = client.post(
            "http://localhost:11434/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": (
                    "Rewrite the following text in your own words, preserving the meaning "
                    "exactly. Keep roughly the same length. Output only the rewritten text."
                    "\n\n" + text
                ),
                "stream": False,
            },
            timeout=180,
        )
    resp.raise_for_status()
    return resp.json()["response"].strip()


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for text_id, base, manual in BASE_TEXTS:
        rows.append(
            {"id": f"{text_id}-verbatim", "text_a": base, "text_b": base, "label": "verbatim"}
        )
        paraphrase = manual if manual else llm_paraphrase(base)
        origin = "manual" if manual else f"llm:{OLLAMA_MODEL}"
        rows.append(
            {
                "id": f"{text_id}-paraphrase",
                "text_a": base,
                "text_b": paraphrase,
                "label": "paraphrase",
                "origin": origin,
            }
        )

    for (base_id, base, _), (unrel_id, unrel) in zip(BASE_TEXTS, UNRELATED * 3, strict=False):
        rows.append(
            {"id": f"{base_id}-vs-{unrel_id}", "text_a": base, "text_b": unrel, "label": "original"}
        )

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts = {}
    for r in rows:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    print(f"Записано {len(rows)} пар в {OUT_PATH}: {counts}")


if __name__ == "__main__":
    main()
