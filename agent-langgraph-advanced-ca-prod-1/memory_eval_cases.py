"""Curated scenarios for the managed-memory evaluation suite.

Each scenario is one isolated story. Turns share a long-term memory scope, while
``new_session=True`` starts a fresh conversation thread. Expectations describe
observable behavior and are never sent to the agent.
"""

from __future__ import annotations

from typing import Any


CATEGORIES = {
    "explicit-save",
    "implicit-save",
    "no-save",
    "update",
    "explicit-search",
    "implicit-search",
    "no-search",
    "multi-turn-lifecycle",
}
WRITE_ACTIONS = {"save", "update", "none"}
SEARCH_EXPECTATIONS = {"required", "forbidden"}


SCENARIOS: list[dict[str, Any]] = [
    {
        "id": "explicit-save-coffee-preference",
        "category": "explicit-save",
        "turns": [
            {
                "message": "Please remember that I always take my coffee black and never add sweetener.",
                "expect": {
                    "write": "save",
                    "search": "required",
                    "memory_should": (
                        "Store the durable preference that the user drinks coffee black and does not "
                        "add sweetener. Do not invent a preferred roast, brewing method, or shop."
                    ),
                },
            },
            {
                "message": "How do I take my coffee?",
                "new_session": True,
                "expect": {
                    "write": "none",
                    "search": "required",
                    "search_should_contain": ["black", "sweetener"],
                    "answer_should": (
                        "State that the user drinks coffee black and does not add sweetener."
                    ),
                },
            },
        ],
    },
    {
        "id": "explicit-save-renovation-brief",
        "category": "explicit-save",
        "turns": [
            {
                "message": (
                    "Remember my bathroom renovation details: the budget is $18,000 CAD, "
                    "Halverson Bros starts October 6, we want a walk-in shower and heated floors, "
                    "we are reusing the vanity, and the target finish is mid-December."
                ),
                "expect": {
                    "write": "save",
                    "search": "required",
                    "memory_should": (
                        "Store one coherent bathroom-renovation memory containing the budget, "
                        "contractor, start date, requested features, reused vanity, and target finish. "
                        "Keep the description short and put the detailed brief in contents."
                    ),
                },
            },
            {
                "message": "Remind me of the renovation budget, contractor, and two new features.",
                "new_session": True,
                "expect": {
                    "write": "none",
                    "search": "required",
                    "search_should_contain": [
                        "18,000",
                        "Halverson",
                        "walk-in shower",
                        "heated floors",
                    ],
                    "answer_should": (
                        "Report the $18,000 CAD budget, Halverson Bros, the walk-in shower, "
                        "and heated floors."
                    ),
                },
            },
        ],
    },
    {
        "id": "implicit-save-editor-workflow",
        "category": "implicit-save",
        "turns": [
            {
                "message": (
                    "For my Python work I use Neovim, and I always run Ruff before I commit. "
                    "Can you explain what Ruff's fix mode does?"
                ),
                "expect": {
                    "write": "save",
                    "search": "required",
                    "memory_should": (
                        "Store the stable workflow preferences that the user uses Neovim for "
                        "Python and runs Ruff before committing. Do not store the generic explanation."
                    ),
                },
            },
            {
                "message": "Give me a quick edit-and-check workflow for this Python change.",
                "new_session": True,
                "expect": {
                    "write": "none",
                    "search": "required",
                    "search_should_contain": ["Neovim", "Ruff"],
                    "answer_should": (
                        "Tailor the workflow around editing in Neovim and running Ruff before commit."
                    ),
                },
            },
        ],
    },
    {
        "id": "implicit-save-meeting-preference",
        "category": "implicit-save",
        "turns": [
            {
                "message": (
                    "I work on Pacific time and avoid meetings before 10 a.m. because mornings are "
                    "reserved for focused work. What is a polite way to decline an 8 a.m. invite?"
                ),
                "expect": {
                    "write": "save",
                    "search": "required",
                    "memory_should": (
                        "Store the durable scheduling preference: Pacific time, no meetings before "
                        "10 a.m., and mornings reserved for focused work."
                    ),
                },
            }
        ],
    },
    {
        "id": "implicit-save-launch-project",
        "category": "implicit-save",
        "turns": [
            {
                "message": (
                    "I'm leading the Atlas launch. The beta is due September 12, Maya owns the "
                    "documentation, and the main unresolved risk is mobile authentication. Help me "
                    "write today's stand-up update."
                ),
                "expect": {
                    "write": "save",
                    "search": "required",
                    "memory_should": (
                        "Store one ongoing Atlas-launch memory with the beta date, Maya's documentation "
                        "ownership, and the mobile-authentication risk. Do not store the drafted update."
                    ),
                },
            },
            {
                "message": "What should I prioritize for the Atlas launch this week?",
                "new_session": True,
                "expect": {
                    "write": "none",
                    "search": "required",
                    "search_should_contain": ["September 12", "Maya", "mobile authentication"],
                    "answer_should": (
                        "Prioritize the September 12 beta and unresolved mobile-authentication risk, "
                        "while recognizing that Maya owns documentation."
                    ),
                },
            },
        ],
    },
    {
        "id": "no-save-temporary-nickname",
        "category": "no-save",
        "turns": [
            {
                "message": "For this conversation only, call me Captain.",
                "expect": {"write": "none", "search": "forbidden"},
            },
            {
                "message": "What name do I prefer you to call me?",
                "new_session": True,
                "expect": {
                    "write": "none",
                    "search": "required",
                    "answer_should": (
                        "Do not claim Captain is a durable preference; explain that no lasting name "
                        "preference is stored."
                    ),
                },
            },
        ],
    },
    {
        "id": "no-save-hypothetical-diet",
        "category": "no-save",
        "turns": [
            {
                "message": "Hypothetically, if I were vegan, what would a simple week of meal prep look like?",
                "expect": {"write": "none"},
            }
        ],
    },
    {
        "id": "no-save-undecided-move",
        "category": "no-save",
        "turns": [
            {
                "message": (
                    "I'm only considering a move to Berlin; I have not decided and may stay where I am. "
                    "What questions should I think through?"
                ),
                "expect": {"write": "none"},
            }
        ],
    },
    {
        "id": "no-save-sensitive-health-without-consent",
        "category": "no-save",
        "turns": [
            {
                "message": "I take lisinopril for blood pressure. What time of day is it usually taken?",
                "expect": {"write": "none"},
            }
        ],
    },
    {
        "id": "update-current-job",
        "category": "update",
        "initial_memories": [
            {
                "path": "/memories/work/current-role.md",
                "description": "Works as a barista at Luna Cafe downtown",
                "contents": "- Role: barista\n- Employer: Luna Cafe\n- Location: downtown",
            }
        ],
        "turns": [
            {
                "message": "I got promoted and am now the shift supervisor at Luna Cafe downtown.",
                "expect": {
                    "write": "update",
                    "search": "required",
                    "memory_should": (
                        "Update the existing role to shift supervisor without creating another memory. "
                        "Preserve Luna Cafe and the downtown location; do not present barista as current."
                    ),
                },
            },
            {
                "message": "What do I currently do for work?",
                "new_session": True,
                "expect": {
                    "write": "none",
                    "search": "required",
                    "search_should_contain": ["shift supervisor", "Luna Cafe", "downtown"],
                    "answer_should": (
                        "State that the user's current role is shift supervisor at Luna Cafe downtown."
                    ),
                },
            },
        ],
    },
    {
        "id": "update-project-with-preservation",
        "category": "update",
        "initial_memories": [
            {
                "path": "/memories/projects/bathroom-renovation.md",
                "description": "Bathroom renovation planned for $18k CAD, finishing mid-December",
                "contents": (
                    "- Budget: $18,000 CAD\n- Contractor: Halverson Bros\n"
                    "- Features: walk-in shower and heated floors\n- Reuse existing vanity\n"
                    "- Target finish: mid-December"
                ),
            }
        ],
        "turns": [
            {
                "message": (
                    "The bathroom renovation budget is now $20,000 CAD and the finish moved to "
                    "January 15. Everything else is unchanged."
                ),
                "expect": {
                    "write": "update",
                    "search": "required",
                    "memory_should": (
                        "Update the existing renovation memory to $20,000 CAD and January 15. "
                        "Preserve Halverson Bros, the shower, heated floors, and reused vanity."
                    ),
                },
            },
            {
                "message": "Give me the current renovation budget, finish date, and contractor.",
                "new_session": True,
                "expect": {
                    "write": "none",
                    "search": "required",
                    "search_should_contain": ["20,000", "January 15", "Halverson"],
                    "answer_should": (
                        "Report the current $20,000 CAD budget, January 15 finish, and Halverson Bros."
                    ),
                },
            },
        ],
    },
    {
        "id": "explicit-search-editor-preference",
        "category": "explicit-search",
        "initial_memories": [
            {
                "path": "/memories/preferences/editor.md",
                "description": "Uses Neovim for Python development",
                "contents": "Prefers Neovim for Python development and uses a minimal plugin setup.",
            }
        ],
        "turns": [
            {
                "message": "Search your memory and tell me which editor I prefer for Python.",
                "expect": {
                    "write": "none",
                    "search": "required",
                    "search_should_contain": ["Neovim", "Python"],
                    "answer_should": "State that the user prefers Neovim for Python development.",
                },
            }
        ],
    },
    {
        "id": "explicit-search-nearby-travel-facts",
        "category": "explicit-search",
        "initial_memories": [
            {
                "path": "/memories/travel/user-flight-preferences.md",
                "description": "User flies from SEA and prefers aisle seats",
                "contents": "- Home airport: SEA\n- Seat preference: aisle",
            },
            {
                "path": "/memories/family/renee-flight-preferences.md",
                "description": "Renee flies from PDX and prefers window seats",
                "contents": "- Person: sister Renee\n- Home airport: PDX\n- Seat preference: window",
            },
        ],
        "turns": [
            {
                "message": "Check your memory: what are my own airport and seat preferences?",
                "expect": {
                    "write": "none",
                    "search": "required",
                    "search_should_contain": ["SEA", "aisle"],
                    "answer_should": (
                        "Say that the user flies from SEA and prefers aisle seats. Do not substitute "
                        "Renee's PDX or window-seat preferences."
                    ),
                },
            }
        ],
    },
    {
        "id": "explicit-search-pet-crowded-scope",
        "category": "explicit-search",
        "initial_memories": [
            {
                "path": "/memories/pets/mochi.md",
                "description": "Mochi is a young golden retriever",
                "contents": (
                    "Mochi is two years old, highly food-motivated, and enjoys scent games."
                ),
            },
            {
                "path": "/memories/family/renee-cat.md",
                "description": "Renee's pet cat prefers indoor window perches",
                "contents": "Renee uses a window perch to entertain her cat on rainy days.",
            },
            {
                "path": "/memories/family/eli-dog.md",
                "description": "Eli's pet dog refuses rainy walks",
                "contents": "Eli gives his dog indoor puzzle toys when it rains.",
            },
            {
                "path": "/memories/friends/morgan-rabbit.md",
                "description": "Morgan's pet rabbit uses an indoor tunnel",
                "contents": "Morgan sets up the tunnel as a rainy-day activity.",
            },
            {
                "path": "/memories/friends/sam-parrot.md",
                "description": "Sam's pet parrot needs indoor enrichment",
                "contents": "Sam rotates foraging toys during rainy weather.",
            },
            {
                "path": "/memories/friends/priya-hamster.md",
                "description": "Priya's pet hamster has an indoor playpen",
                "contents": "The playpen is Priya's preferred rainy-day setup.",
            },
            {
                "path": "/memories/friends/jules-guinea-pig.md",
                "description": "Jules keeps a pet guinea pig indoors",
                "contents": "Jules uses cardboard tunnels for indoor enrichment.",
            },
            {
                "path": "/memories/work/office-fish.md",
                "description": "Office pet fish remain indoors",
                "contents": "The office aquarium belongs to the workplace, not the user.",
            },
            {
                "path": "/memories/neighbors/terrier.md",
                "description": "Neighbor's pet terrier likes rainy-day games",
                "contents": "The neighbor plays indoor tug with the terrier.",
            },
            {
                "path": "/memories/services/pet-sitter.md",
                "description": "Pet sitter offers indoor rainy-day visits",
                "contents": "This is a service note, not a fact about the user's animal.",
            },
            {
                "path": "/memories/volunteering/animal-shelter.md",
                "description": "Animal shelter needs indoor pet volunteers on rainy days",
                "contents": "This is a volunteering opportunity, not the user's pet routine.",
            },
            {
                "path": "/memories/local/rainy-dog-park.md",
                "description": "Indoor pet center opens when the dog park is rainy",
                "contents": "This local-business note does not identify the user's animal.",
            },
            {
                "path": "/memories/shopping/pet-supplies.md",
                "description": "Indoor pet supplies for rainy weekends",
                "contents": "Generic shopping list: balls, tunnels, and puzzle toys.",
            },
        ],
        "turns": [
            {
                "message": (
                    "Search your memory and tell me which indoor activity would work best for my "
                    "own pet on a rainy day."
                ),
                "expect": {
                    "write": "none",
                    "search": "required",
                    "search_should_contain": ["Mochi", "golden retriever", "food-motivated"],
                    "answer_should": (
                        "Recommend a food-based scent game for Mochi, the user's young "
                        "food-motivated golden retriever. Do not substitute another person's animal "
                        "or a generic service, volunteering, local-business, or shopping memory."
                    ),
                },
            }
        ],
    },
    {
        "id": "implicit-search-sushi-order",
        "category": "implicit-search",
        "initial_memories": [
            {
                "path": "/memories/preferences/food.md",
                "description": "Vegetarian with a severe shellfish allergy",
                "contents": (
                    "- Diet: strictly vegetarian\n- Allergy: severe shellfish allergy\n"
                    "- Cross-contamination matters"
                ),
            }
        ],
        "turns": [
            {
                "message": "I'm at a sushi restaurant right now. Order a safe dinner for me.",
                "expect": {
                    "write": "none",
                    "search": "required",
                    "search_should_contain": ["vegetarian", "shellfish", "cross-contamination"],
                    "answer_should": (
                        "Recommend a vegetarian order and explicitly account for the severe shellfish "
                        "allergy and cross-contamination risk."
                    ),
                },
            }
        ],
    },
    {
        "id": "implicit-search-pet-lexical-gap",
        "category": "implicit-search",
        "initial_memories": [
            {
                "path": "/memories/pets/mochi.md",
                "description": "Mochi is a young golden retriever",
                "contents": (
                    "Mochi is a 2-year-old golden retriever who is highly food-motivated and "
                    "usually takes a long seawall walk each morning."
                ),
            }
        ],
        "turns": [
            {
                "message": "How can I keep my pet entertained indoors on a rainy morning?",
                "expect": {
                    "write": "none",
                    "search": "required",
                    "search_should_contain": ["Mochi", "golden retriever", "food-motivated"],
                    "answer_should": (
                        "Give rainy-day ideas tailored to Mochi, a young food-motivated golden retriever, "
                        "rather than generic advice for an unknown pet."
                    ),
                },
            }
        ],
    },
    {
        "id": "implicit-search-friday-synthesis",
        "category": "implicit-search",
        "initial_memories": [
            {
                "path": "/memories/preferences/budget.md",
                "description": "Keeping discretionary spending low this year",
                "contents": "Prefers inexpensive plans while saving aggressively this year.",
            },
            {
                "path": "/memories/profile/location.md",
                "description": "Lives in Austin, Texas",
                "contents": "Home city: Austin, Texas.",
            },
            {
                "path": "/memories/preferences/food.md",
                "description": "Eats strictly gluten-free",
                "contents": "Dietary requirement: strictly gluten-free.",
            },
        ],
        "turns": [
            {
                "message": "Plan my Friday evening.",
                "expect": {
                    "write": "none",
                    "search": "required",
                    "search_should_contain": ["Austin", "gluten-free", "inexpensive"],
                    "answer_should": (
                        "Propose an inexpensive Friday-evening plan in Austin with a specifically "
                        "gluten-free food option."
                    ),
                },
            }
        ],
    },
    {
        "id": "implicit-search-atlas-name-collision",
        "category": "implicit-search",
        "initial_memories": [
            {
                "path": "/memories/work/atlas-launch.md",
                "description": "Atlas product beta is due September 12",
                "contents": (
                    "- User leads the Atlas product launch\n- Beta due: September 12\n"
                    "- Documentation owner: Maya\n- Main unresolved risk: mobile authentication"
                ),
            },
            {
                "path": "/memories/travel/atlas-mountains.md",
                "description": "Atlas Mountains hiking trip planned for November",
                "contents": "Guide: Karim. The trip is a personal vacation in Morocco.",
            },
            {
                "path": "/memories/reading/historical-atlases.md",
                "description": "Reading a book about historical atlases",
                "contents": "Current leisure reading covers the history of mapmaking.",
            },
        ],
        "turns": [
            {
                "message": "What should I focus on this week to keep the Atlas beta on track?",
                "expect": {
                    "write": "none",
                    "search": "required",
                    "search_should_contain": ["September 12", "Maya", "mobile authentication"],
                    "answer_should": (
                        "Prioritize the September 12 product beta and mobile-authentication risk, "
                        "while recognizing that Maya owns documentation. Do not confuse the product "
                        "with the Atlas Mountains trip or the mapmaking book."
                    ),
                },
            }
        ],
    },
    {
        "id": "implicit-search-family-food-disambiguation",
        "category": "implicit-search",
        "initial_memories": [
            {
                "path": "/memories/preferences/user-food.md",
                "description": "User eats strictly gluten-free",
                "contents": "Dietary requirement: strictly gluten-free.",
            },
            {
                "path": "/memories/family/renee-food.md",
                "description": "Renee has a severe shellfish allergy",
                "contents": (
                    "Sister Renee has a severe shellfish allergy. Cross-contamination matters. "
                    "Renee is not vegetarian."
                ),
            },
            {
                "path": "/memories/family/eli-food.md",
                "description": "Partner Eli is vegan",
                "contents": "Eli is vegan and has no known food allergies.",
            },
            {
                "path": "/memories/friends/morgan-food.md",
                "description": "Morgan dislikes spicy food",
                "contents": "Friend Morgan prefers mild food but has no dietary restrictions.",
            },
        ],
        "turns": [
            {
                "message": "Renee is visiting tonight. Pick a restaurant order for us.",
                "expect": {
                    "write": "none",
                    "search": "required",
                    "search_should_contain": [
                        "gluten-free",
                        "Renee",
                        "shellfish",
                        "cross-contamination",
                    ],
                    "answer_should": (
                        "Recommend a concrete gluten-free and shellfish-free order for the user and "
                        "Renee, explicitly accounting for cross-contamination. Do not apply Eli's "
                        "vegan preference or Morgan's spice preference to them."
                    ),
                },
            }
        ],
    },
    {
        "id": "implicit-search-current-commute-over-history",
        "category": "implicit-search",
        "initial_memories": [
            {
                "path": "/memories/routines/current-commute.md",
                "description": "Cycles to work on Tuesdays and Thursdays",
                "contents": "Current commute: bicycle on Tuesday and Thursday.",
            },
            {
                "path": "/memories/history/old-commute.md",
                "description": "Previously took Caltrain before moving",
                "contents": "Historical only: the user no longer takes Caltrain to work.",
            },
            {
                "path": "/memories/family/partner-commute.md",
                "description": "Partner drives to work",
                "contents": "The user's partner commutes by car every weekday.",
            },
            {
                "path": "/memories/preferences/intercity-travel.md",
                "description": "Prefers trains for intercity travel",
                "contents": "For travel between cities, the user prefers trains over flying.",
            },
        ],
        "turns": [
            {
                "message": "I have an unusually early meeting this Thursday. How should I adjust my commute?",
                "expect": {
                    "write": "none",
                    "search": "required",
                    "search_should_contain": ["bicycle", "Thursday"],
                    "answer_should": (
                        "Adapt the user's current Thursday bicycle commute, such as leaving earlier "
                        "and preparing equipment or checking conditions. Do not assume the user still "
                        "takes Caltrain or uses the partner's car."
                    ),
                },
            }
        ],
    },
    {
        "id": "implicit-search-python-workflow-among-distractors",
        "category": "implicit-search",
        "initial_memories": [
            {
                "path": "/memories/workflows/python.md",
                "description": "Python work uses Neovim and Ruff",
                "contents": "Edit Python in Neovim and always run Ruff before committing.",
            },
            {
                "path": "/memories/workflows/go.md",
                "description": "Go work uses VS Code and golangci-lint",
                "contents": "For Go, edit in VS Code and run golangci-lint.",
            },
            {
                "path": "/memories/workflows/frontend.md",
                "description": "Frontend work uses WebStorm, ESLint, and Prettier",
                "contents": "For frontend changes, use WebStorm, ESLint, and Prettier.",
            },
            {
                "path": "/memories/workflows/general-code-review.md",
                "description": "Runs unit tests before opening a pull request",
                "contents": "General workflow preference: run unit tests before opening a pull request.",
            },
        ],
        "turns": [
            {
                "message": "Give me my usual edit-and-check sequence for this Python hotfix.",
                "expect": {
                    "write": "none",
                    "search": "required",
                    "search_should_contain": ["Neovim", "Ruff", "unit tests"],
                    "answer_should": (
                        "Tailor the sequence to editing Python in Neovim, running Ruff before commit, "
                        "and running unit tests before opening a pull request. Do not substitute the "
                        "Go or frontend tools."
                    ),
                },
            }
        ],
    },
    {
        "id": "implicit-search-renovation-budget-calculation",
        "category": "implicit-search",
        "initial_memories": [
            {
                "path": "/memories/projects/current-renovation-budget.md",
                "description": "Current renovation budget is $20,000 CAD",
                "contents": "- Current budget: $20,000 CAD\n- Amount already committed: $17,900 CAD",
            },
            {
                "path": "/memories/history/old-renovation-estimate.md",
                "description": "Superseded renovation estimate was $18,000 CAD",
                "contents": "Historical only: the $18,000 CAD estimate is no longer current.",
            },
            {
                "path": "/memories/projects/wedding-budget.md",
                "description": "Wedding budget is $20,000 CAD",
                "contents": "Separate project: wedding budget is $20,000 CAD.",
            },
            {
                "path": "/memories/budgets/furniture.md",
                "description": "Furniture allowance is $1,500 CAD",
                "contents": "Separate furniture allowance: $1,500 CAD.",
            },
        ],
        "turns": [
            {
                "message": "Can I add a $1,500 fixture without exceeding the current renovation budget?",
                "expect": {
                    "write": "none",
                    "search": "required",
                    "search_should_contain": ["20,000", "17,900"],
                    "answer_should": (
                        "Use the current renovation figures to calculate $19,400 CAD committed after "
                        "the fixture, which is $600 below the $20,000 CAD budget. Do not use the "
                        "superseded $18,000 estimate or either unrelated budget."
                    ),
                },
            }
        ],
    },
    {
        "id": "implicit-search-weekend-outing-lexical-gap",
        "category": "implicit-search",
        "initial_memories": [
            {
                "path": "/memories/routines/sunday-hiking.md",
                "description": "Hikes Mount Si on Sunday mornings",
                "contents": (
                    "The user hikes Mount Si on Sunday mornings for climbing endurance and is "
                    "training for an Enchantments through-hike in August."
                ),
            },
            {
                "path": "/memories/routines/gym.md",
                "description": "Goes to the gym Wednesday evenings",
                "contents": "Usual gym session is Wednesday evening and focuses on upper-body strength.",
            },
            {
                "path": "/memories/friends/morgan-hiking.md",
                "description": "Morgan hikes Rattlesnake Ledge on Saturdays",
                "contents": "Friend Morgan usually hikes Rattlesnake Ledge on Saturday mornings.",
            },
            {
                "path": "/memories/routines/sunday-meal-prep.md",
                "description": "Meal preps on Sunday evenings",
                "contents": "The user's Sunday-evening routine is meal preparation for the week.",
            },
        ],
        "turns": [
            {
                "message": (
                    "The trail is closed this Sunday. Give me an indoor substitute that still feels "
                    "like my usual morning outing."
                ),
                "expect": {
                    "write": "none",
                    "search": "required",
                    "search_should_contain": ["Mount Si", "Sunday mornings", "Enchantments"],
                    "answer_should": (
                        "Propose a Sunday-morning indoor climbing-endurance workout, such as stairs "
                        "or an incline treadmill, that supports the user's Enchantments training. Do "
                        "not substitute Morgan's Saturday hike, Wednesday gym routine, or meal prep."
                    ),
                },
            }
        ],
    },
    {
        "id": "implicit-search-headache-trigger-semantic-gap",
        "category": "implicit-search",
        "initial_memories": [
            {
                "path": "/memories/preferences/work-environment.md",
                "description": "Environmental conditions that prevent migraines while working",
                "contents": (
                    "Fluorescent lighting and strong fragrances are reliable migraine triggers for "
                    "the user. The user works best in a quiet space with natural or warm lighting."
                ),
            },
            {
                "path": "/memories/preferences/desk.md",
                "description": "Prefers a standing desk for long work sessions",
                "contents": "The user alternates between sitting and standing while working.",
            },
        ],
        "turns": [
            {
                "message": (
                    "Pick the kind of place where I can work for three hours without ending up "
                    "with a pounding head."
                ),
                "expect": {
                    "write": "none",
                    "search": "required",
                    "search_should_contain": ["fluorescent", "fragrances", "warm lighting"],
                    "answer_should": (
                        "Recommend a quiet workspace with natural or warm lighting and without "
                        "strong fragrances, explicitly connecting those constraints to avoiding "
                        "the user's known migraine triggers."
                    ),
                },
            }
        ],
    },
    {
        "id": "implicit-search-twins-name-gap",
        "category": "implicit-search",
        "initial_memories": [
            {
                "path": "/memories/family/nora-wednesday.md",
                "description": "Nora's Wednesday after-school schedule",
                "contents": (
                    "Nora leaves Cedar Elementary at 3:10 p.m. on Wednesdays and must be taken "
                    "directly to violin at 4:00 p.m."
                ),
            },
            {
                "path": "/memories/family/leo-wednesday.md",
                "description": "Leo's Wednesday after-school schedule",
                "contents": (
                    "Leo stays in school aftercare on Wednesdays and can be collected any time "
                    "before 5:30 p.m."
                ),
            },
            {
                "path": "/memories/family/renee-wednesday.md",
                "description": "Renee attends a Wednesday evening pottery class",
                "contents": "Renee's class is unrelated to the children's school schedule.",
            },
        ],
        "turns": [
            {
                "message": "How should I handle the twins after school this Wednesday?",
                "expect": {
                    "write": "none",
                    "search": "required",
                    "search_should_contain": [
                        "Nora",
                        "3:10",
                        "violin",
                        "Leo",
                        "5:30",
                    ],
                    "answer_should": (
                        "Give a workable plan for both children: collect Nora at Cedar Elementary "
                        "at 3:10 p.m. and take her to violin by 4:00 p.m.; collect Leo from "
                        "aftercare by 5:30 p.m. Do not substitute Renee's pottery class."
                    ),
                },
            }
        ],
    },
    {
        "id": "implicit-search-rainy-commute-multihop",
        "category": "implicit-search",
        "initial_memories": [
            {
                "path": "/memories/routines/bad-weather-commute.md",
                "description": "Fallback plan for severe-weather office travel",
                "contents": "The user's bad-weather commute is called the Bluebird plan.",
            },
            {
                "path": "/memories/routines/bluebird.md",
                "description": "Bluebird route details",
                "contents": (
                    "For the Bluebird plan, take Route 14 from 7th and Pine at 7:35 a.m., then "
                    "walk two blocks from the final stop to the office."
                ),
            },
            {
                "path": "/memories/routines/normal-commute.md",
                "description": "Normal commute is by bicycle",
                "contents": "In ordinary weather the user cycles to the office.",
            },
        ],
        "turns": [
            {
                "message": "It's pouring tomorrow morning. How should I get to the office?",
                "expect": {
                    "write": "none",
                    "search": "required",
                    "search_should_contain": ["Bluebird", "Route 14", "7th and Pine", "7:35"],
                    "answer_should": (
                        "Use the bad-weather Bluebird plan: take Route 14 from 7th and Pine at "
                        "7:35 a.m. and walk two blocks from the final stop. Do not recommend the "
                        "normal bicycle commute."
                    ),
                },
            }
        ],
    },
    {
        "id": "implicit-search-red-eye-seat-exception",
        "category": "implicit-search",
        "initial_memories": [
            {
                "path": "/memories/preferences/flight-seats.md",
                "description": "User's flight seat preferences and exceptions",
                "contents": (
                    "The user usually chooses an aisle seat. For overnight flights, the exception "
                    "is a window seat on the right side, away from the galley, so the user can sleep."
                ),
            },
            {
                "path": "/memories/family/renee-flight-seats.md",
                "description": "Renee always chooses an aisle seat",
                "contents": "Renee's seat preference does not apply to the user.",
            },
        ],
        "turns": [
            {
                "message": "I'm booking tonight's red-eye. Which seat should I pick?",
                "expect": {
                    "write": "none",
                    "search": "required",
                    "search_should_contain": ["overnight", "window", "right side", "galley"],
                    "answer_should": (
                        "Apply the user's overnight-flight exception: choose a right-side window "
                        "seat away from the galley. Do not apply the usual aisle preference or "
                        "Renee's preference."
                    ),
                },
            }
        ],
    },
    {
        "id": "implicit-search-locked-out-access-gap",
        "category": "implicit-search",
        "initial_memories": [
            {
                "path": "/memories/home/emergency-entry.md",
                "description": "Home access contingency",
                "contents": (
                    "Emergency entry uses the blue magnetic lockbox behind the east downspout. "
                    "The combination is intentionally not stored in memory."
                ),
            },
            {
                "path": "/memories/family/eli-spare-key.md",
                "description": "Eli keeps a spare key for his own apartment",
                "contents": "Eli's spare key does not open the user's home.",
            },
        ],
        "turns": [
            {
                "message": "I shut the door with my keys still inside. How do I get back in?",
                "expect": {
                    "write": "none",
                    "search": "required",
                    "search_should_contain": ["blue", "east downspout", "intentionally not stored"],
                    "answer_should": (
                        "Direct the user to the blue magnetic lockbox behind the east downspout "
                        "and explain that its combination is not stored. Do not invent a code or "
                        "suggest using Eli's unrelated spare key."
                    ),
                },
            }
        ],
    },
    {
        "id": "implicit-search-mentor-diet-multihop",
        "category": "implicit-search",
        "initial_memories": [
            {
                "path": "/memories/work/mei-chen-relationship.md",
                "description": "Career relationship with Dr. Mei Chen",
                "contents": (
                    "Dr. Mei Chen is the user's former manager and sponsored the user's promotion."
                ),
            },
            {
                "path": "/memories/people/mei-chen-food.md",
                "description": "Mei Chen's food requirement",
                "contents": (
                    "Mei Chen follows a strict gluten-free diet. Baked goods for her must be "
                    "certified gluten-free and prepared using dedicated equipment."
                ),
            },
            {
                "path": "/memories/people/morgan-food.md",
                "description": "Morgan enjoys almond croissants",
                "contents": "Morgan's pastry preference is unrelated to Mei Chen.",
            },
        ],
        "turns": [
            {
                "message": (
                    "I'm bringing pastries to the mentor who helped me get promoted. Is there "
                    "anything I should account for?"
                ),
                "expect": {
                    "write": "none",
                    "search": "required",
                    "search_should_contain": [
                        "Mei Chen",
                        "strict gluten-free",
                        "certified gluten-free",
                        "dedicated equipment",
                    ],
                    "answer_should": (
                        "Resolve the mentor as Dr. Mei Chen and recommend only certified "
                        "gluten-free pastries prepared using dedicated equipment. Do not apply "
                        "Morgan's pastry preference."
                    ),
                },
            }
        ],
    },
    {
        "id": "implicit-search-instrument-lexical-gap",
        "category": "implicit-search",
        "initial_memories": [
            {
                "path": "/memories/hobbies/jay-haide.md",
                "description": "Care routine for the user's 2016 Jay Haide cello",
                "contents": (
                    "Keep the cello in a hard case, maintain 40-55% humidity, loosen the bow hair "
                    "after playing, and never gate-check it."
                ),
            },
            {
                "path": "/memories/travel/luggage.md",
                "description": "Flight luggage preference",
                "contents": "The user usually travels with one small carry-on roller bag.",
            },
            {
                "path": "/memories/family/eli-violin.md",
                "description": "Eli owns a student violin",
                "contents": "Eli's violin and its care routine do not belong to the user.",
            },
        ],
        "turns": [
            {
                "message": "I'm flying next week. How should I protect my instrument in transit?",
                "expect": {
                    "write": "none",
                    "search": "required",
                    "search_should_contain": [
                        "cello",
                        "hard case",
                        "40-55% humidity",
                        "never gate-check",
                    ],
                    "answer_should": (
                        "Recognize the user's instrument as the 2016 Jay Haide cello and advise a "
                        "hard case, 40-55% humidity, loosened bow hair, and never gate-checking it. "
                        "Do not substitute Eli's violin or discuss only ordinary luggage."
                    ),
                },
            }
        ],
    },
    {
        "id": "implicit-search-houseplant-colloquial-gap",
        "category": "implicit-search",
        "initial_memories": [
            {
                "path": "/memories/home/calathea-orbifolia.md",
                "description": "Calathea orbifolia care routine",
                "contents": (
                    "Use filtered water, keep humidity at or above 60%, and avoid direct "
                    "afternoon sun."
                ),
            },
            {
                "path": "/memories/family/renee-pothos.md",
                "description": "Renee's houseplant has occasional brown leaves",
                "contents": "Renee waters her pothos with tap water; this is not the user's plant.",
            },
            {
                "path": "/memories/home/thermostat.md",
                "description": "Keeps the apartment at 68 degrees",
                "contents": "The user prefers a cool apartment.",
            },
        ],
        "turns": [
            {
                "message": "My leafy roommate has crispy brown edges again. What should I change?",
                "expect": {
                    "write": "none",
                    "search": "required",
                    "search_should_contain": [
                        "Calathea orbifolia",
                        "filtered water",
                        "60%",
                        "afternoon sun",
                    ],
                    "answer_should": (
                        "Recognize the user's Calathea orbifolia and recommend filtered water, at "
                        "least 60% humidity, and avoiding direct afternoon sun. Do not apply "
                        "Renee's pothos routine."
                    ),
                },
            }
        ],
    },
    {
        "id": "no-search-calculus",
        "category": "no-search",
        "turns": [
            {
                "message": "What is the derivative of x^3 + 2x?",
                "expect": {"write": "none", "search": "forbidden"},
            }
        ],
    },
    {
        "id": "no-search-python-closures",
        "category": "no-search",
        "turns": [
            {
                "message": "Explain Python closures in two sentences.",
                "expect": {"write": "none", "search": "forbidden"},
            }
        ],
    },
    {
        "id": "lifecycle-explicit-save-new-session-recall",
        "category": "multi-turn-lifecycle",
        "turns": [
            {
                "message": (
                    "Remember my hiking setup: I hike Mount Si on Sunday mornings, wear Salomon "
                    "X Ultra boots, and I am training for an Enchantments through-hike in August."
                ),
                "expect": {
                    "write": "save",
                    "search": "required",
                    "memory_should": (
                        "Store one hiking memory with Mount Si on Sunday mornings, Salomon X Ultra "
                        "boots, and the August Enchantments through-hike goal."
                    ),
                },
            },
            {
                "message": "What boots do I hike in, and what am I training for?",
                "new_session": True,
                "expect": {
                    "write": "none",
                    "search": "required",
                    "search_should_contain": ["Salomon X Ultra", "Enchantments", "August"],
                    "answer_should": (
                        "State that the user wears Salomon X Ultra boots and is training for an "
                        "August Enchantments through-hike."
                    ),
                },
            },
        ],
    },
    {
        "id": "lifecycle-save-update-latest-wins",
        "category": "multi-turn-lifecycle",
        "turns": [
            {
                "message": "Remember that the Orion beta launches October 10 and Priya owns testing.",
                "expect": {
                    "write": "save",
                    "search": "required",
                    "memory_should": (
                        "Store one Orion beta memory with the October 10 launch and Priya owning testing."
                    ),
                },
            },
            {
                "message": (
                    "The Orion beta moved to October 24. Priya still owns testing, and add that "
                    "the release is blocked on SSO validation."
                ),
                "new_session": True,
                "expect": {
                    "write": "update",
                    "search": "required",
                    "memory_should": (
                        "Update the same Orion memory to October 24, preserve Priya's testing "
                        "ownership, and add the SSO-validation blocker. October 10 is no longer current."
                    ),
                },
            },
            {
                "message": "Give me the current Orion launch date, testing owner, and blocker.",
                "new_session": True,
                "expect": {
                    "write": "none",
                    "search": "required",
                    "search_should_contain": ["October 24", "Priya", "SSO validation"],
                    "answer_should": (
                        "Report October 24 as the current date, Priya as testing owner, and SSO "
                        "validation as the blocker. Do not present October 10 as current."
                    ),
                },
            },
        ],
    },
]


def validate_scenarios(scenarios: list[dict[str, Any]] = SCENARIOS) -> None:
    """Fail fast when a case is malformed or accidentally ambiguous."""

    seen_ids: set[str] = set()
    for scenario in scenarios:
        scenario_id = scenario.get("id")
        if not isinstance(scenario_id, str) or not scenario_id:
            raise ValueError("Every scenario needs a non-empty string id.")
        if scenario_id in seen_ids:
            raise ValueError(f"Duplicate scenario id: {scenario_id}")
        seen_ids.add(scenario_id)

        category = scenario.get("category")
        if category not in CATEGORIES:
            raise ValueError(f"{scenario_id}: unknown category {category!r}")

        for memory in scenario.get("initial_memories", []):
            path = memory.get("path", "")
            if not path.startswith("/memories/") or not path.endswith(".md"):
                raise ValueError(f"{scenario_id}: invalid initial memory path {path!r}")
            description = memory.get("description")
            if not isinstance(description, str) or not description.strip():
                raise ValueError(f"{scenario_id}: initial memories need descriptions")

        turns = scenario.get("turns")
        if not isinstance(turns, list) or not turns:
            raise ValueError(f"{scenario_id}: at least one turn is required")
        for turn_index, turn in enumerate(turns):
            message = turn.get("message")
            if not isinstance(message, str) or not message.strip():
                raise ValueError(f"{scenario_id} turn {turn_index}: message is required")
            if "new_session" in turn and not isinstance(turn["new_session"], bool):
                raise ValueError(f"{scenario_id} turn {turn_index}: new_session must be boolean")

            expectation = turn.get("expect")
            if not isinstance(expectation, dict):
                raise ValueError(f"{scenario_id} turn {turn_index}: expect is required")
            if expectation.get("write") not in WRITE_ACTIONS:
                raise ValueError(
                    f"{scenario_id} turn {turn_index}: write must be one of {sorted(WRITE_ACTIONS)}"
                )
            search = expectation.get("search")
            if search is not None and search not in SEARCH_EXPECTATIONS:
                raise ValueError(
                    f"{scenario_id} turn {turn_index}: search must be required or forbidden"
                )
            facts = expectation.get("search_should_contain")
            if facts is not None:
                if search != "required" or not isinstance(facts, list) or not facts:
                    raise ValueError(
                        f"{scenario_id} turn {turn_index}: search facts require a required search"
                    )
                if not all(isinstance(fact, str) and fact.strip() for fact in facts):
                    raise ValueError(f"{scenario_id} turn {turn_index}: invalid search fact")
            memory_should = expectation.get("memory_should")
            if memory_should is not None and expectation["write"] == "none":
                raise ValueError(
                    f"{scenario_id} turn {turn_index}: memory_should requires save or update"
                )
            for field in ("memory_should", "answer_should"):
                value = expectation.get(field)
                if value is not None and (not isinstance(value, str) or not value.strip()):
                    raise ValueError(f"{scenario_id} turn {turn_index}: {field} must be text")


validate_scenarios()
