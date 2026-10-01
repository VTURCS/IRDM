"""
Prototype dataset (v1) - the ORIGINAL synthetic data, extracted verbatim.

This is a pure relocation of what used to be hard-coded inside
`research_matching_pipeline.py`. Values are byte-identical to the previously
verified run, so metrics computed against this dataset are unchanged.

    corpus   : 8 listings, VT-001 .. VT-008
    queries  : 4 short, vague student self-descriptions (P1..P4)
    labels   : hand-graded 0/1/2 against the full corpus

Why this is kept as a separate dataset rather than deleted: it is the version
whose numbers are already recorded in the README, and it uses *short vague*
queries, which is the hard case the study is about. The `synthetic_v2` dataset
uses much longer profiles and is a different experiment.

Field mapping to the canonical schema in `data/schema.py`:
    id              -> Opportunity.id
    title           -> Opportunity.title
    description     -> Opportunity.description
    required_skills -> Opportunity.skills
    eligibility     -> Opportunity.eligibility   (metadata only, never filtered)
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from data.schema import Dataset, Opportunity, Query

# -----------------------------------------------------------------------------
# CorpUS: Virginia Tech research opportunity listings (SYNTHETIC)
# -----------------------------------------------------------------------------
# These resemble real VT research postings so the pipeline is executable before
# the real opportunity board is scraped. They are meant to be swapped out.

_RAW_OPPORTUNITIES: List[Dict[str, object]] = [
    {
        "id": "VT-001",
        "title": "Vision-Language Models for Autonomous Driving",
        "description": (
            "Build multimodal perception models that fuse camera imagery with "
            "natural-language scene descriptions to improve pedestrian and "
            "hazard detection for campus autonomous shuttles. Work spans "
            "dataset curation, transformer fine-tuning and closed-loop "
            "simulation evaluation."
        ),
        "required_skills": ["Python", "PyTorch", "Computer Vision", "Deep Learning"],
        "eligibility": "Computer Science, Electrical Engineering, Junior+",
    },
    {
        "id": "VT-002",
        "title": "Smart Grid Load Forecasting with Time-Series ML",
        "description": (
            "Develop forecasting models for campus and regional electricity "
            "demand using historical sensor streams. Projects include feature "
            "engineering for weather data, probabilistic forecasting, and "
            "deployment of dashboards for utility partners."
        ),
        "required_skills": ["Python", "Pandas", "Machine Learning", "Time Series"],
        "eligibility": "Electrical Engineering, Industrial Systems Engineering, Sophomore+",
    },
    {
        "id": "VT-003",
        "title": "Human-Robot Interaction for Assistive Robotics",
        "description": (
            "Design and evaluate collaborative robot behaviors that help older "
            "adults with daily tasks. Involves ROS-based control code, "
            "motion planning, and user studies measuring trust and workload."
        ),
        "required_skills": ["ROS", "C++", "Python", "Robotics"],
        "eligibility": "Mechanical Engineering, Computer Science, Junior+",
    },
    {
        "id": "VT-004",
        "title": "Genomic Data Pipelines for Precision Agriculture",
        "description": (
            "Analyze plant and livestock genomic datasets to identify markers "
            "linked to drought resilience. Tasks include variant calling, "
            "statistical association testing and reproducible pipeline "
            "engineering on high-performance computing clusters."
        ),
        "required_skills": ["R", "Bioinformatics", "Statistics", "Genomics"],
        "eligibility": "Biology, Biochemistry, Computer Science, Sophomore+",
    },
    {
        "id": "VT-005",
        "title": "Usable Security: Evaluating Password Manager UX",
        "description": (
            "Run mixed-methods human subjects studies on how non-experts adopt "
            "password managers. Involves survey and interview design, "
            "qualitative coding, usability testing and lightweight analysis "
            "scripts for behavioral logs."
        ),
        "required_skills": ["Human-Computer Interaction", "Survey Design", "Python", "Statistics"],
        "eligibility": "Computer Science, Psychology, Freshman+",
    },
    {
        "id": "VT-006",
        "title": "Reinforcement Learning for Drone Swarm Coordination",
        "description": (
            "Train multi-agent reinforcement learning policies that let aerial "
            "robot teams survey disaster zones cooperatively. Emphasis on "
            "reward shaping, high-fidelity simulation and sim-to-real transfer."
        ),
        "required_skills": ["Python", "Reinforcement Learning", "PyTorch", "Simulation"],
        "eligibility": "Computer Science, Aerospace Engineering, Junior+",
    },
    {
        "id": "VT-007",
        "title": "LLM-Based Tutoring Chatbot for CS1 Courses",
        "description": (
            "Build and evaluate a retrieval-augmented large language model "
            "tutor that gives programming hints without revealing solutions. "
            "Work covers prompt engineering, natural language processing "
            "evaluation, and a small web front end for students."
        ),
        "required_skills": ["NLP", "Transformers", "Python", "Web Development"],
        "eligibility": "Computer Science, Sophomore+",
    },
    {
        "id": "VT-008",
        "title": "Thermal Management for Electric Vehicle Battery Packs",
        "description": (
            "Model heat generation and cooling strategies for lithium-ion "
            "battery modules. Includes finite element thermal simulation, "
            "experimental validation on instrumented test benches and design "
            "optimization of cold plates."
        ),
        "required_skills": ["MATLAB", "Finite Element Analysis", "Thermodynamics", "Mechanical Design"],
        "eligibility": "Mechanical Engineering, Chemical Engineering, Junior+",
    },
]

# -----------------------------------------------------------------------------
# QUERIES: short, vague undergraduate profiles
# -----------------------------------------------------------------------------
# Deliberately short and underspecified - this is the hard case we care about.
# Each is a 1-2 sentence self-description of skill level, major and interest.

_RAW_PROFILES: List[Dict[str, str]] = [
    {
        "profile_id": "P1",
        "major": "Computer Science",
        "year": "Sophomore",
        "query": "I like machine learning and I know some Python. I want to work on AI stuff.",
    },
    {
        "profile_id": "P2",
        "major": "Biology",
        "year": "Junior",
        "query": "Biology major interested in data and health. I don't code very much.",
    },
    {
        "profile_id": "P3",
        "major": "Mechanical Engineering",
        "year": "Junior",
        "query": "I have built robots before, I like hardware and control systems.",
    },
    {
        "profile_id": "P4",
        "major": "Psychology",
        "year": "Sophomore",
        "query": "I am an HCI and design person, I like studying people and technology.",
    },
]

# -----------------------------------------------------------------------------
# GROUND TRUTH: hand-graded relevance
# -----------------------------------------------------------------------------
# Rubric:
#     0 = Unsuitable      -> topic/skills clearly unrelated to the profile
#     1 = Partial Match   -> adjacent topic or transferable skills; would need
#                            ramp-up or co-mentoring
#     2 = Strong Match    -> profile's stated interests and skills line up with
#                            the listing's core work
#
# IMPORTANT: labels are assigned from listing CONTENT vs profile CONTENT only.
# `eligibility` is deliberately ignored (metadata filtering is OFF), so a
# Sophomore can legitimately be labelled 2 for a "Junior+" listing. This keeps
# the label set aligned with what the retrievers can actually observe.
#
# Grades are written in the SAME ORDER as the corpus so they are easy to audit.

_LABEL_ORDER_NOTE = "grades follow the order of _RAW_OPPORTUNITIES (VT-001 .. VT-008)"

_RELEVANCE_LABELS: Dict[str, List[int]] = {
    # P1: ML + Python, wants AI.
    #     Strong on the ML/AI listings (001, 006, 007); partial where ML is a
    #     supporting tool (002, 005); unsuitable for biology/mechanical work.
    "P1": [2, 1, 0, 0, 1, 2, 2, 0],
    # P2: biology/data/health, minimal coding.
    #     Strong only on genomics (004); partial on applied data work (002) and
    #     the health-adjacent human study (005).
    "P2": [0, 1, 0, 2, 1, 0, 0, 0],
    # P3: robotics/hardware/control.
    #     Strong on HRI robotics (003); partial on autonomous driving (001),
    #     drone swarms (006) and the mechanical/thermal build (008).
    "P3": [1, 0, 2, 0, 0, 1, 0, 1],
    # P4: HCI/design, people + technology.
    #     Strong on usable-security UX (005); partial on human-robot
    #     interaction (003) and the student-facing tutor (007).
    "P4": [0, 0, 1, 0, 2, 0, 1, 0],
}


def _build_relevance_map(
    opportunities: List[Opportunity],
    profiles: List[Query],
) -> Dict[Tuple[str, str], int]:
    """Expand the compact label matrix into {(profile_id, opportunity_id): grade}."""
    if len(_RAW_OPPORTUNITIES) != len(opportunities):
        raise ValueError("Opportunity count changed without updating _RELEVANCE_LABELS.")
    if len(_RAW_PROFILES) != len(profiles):
        raise ValueError("Profile count changed without updating _RELEVANCE_LABELS.")

    mapping: Dict[Tuple[str, str], int] = {}
    for profile in profiles:
        grades = _RELEVANCE_LABELS[profile.profile_id]
        if len(grades) != len(opportunities):
            raise ValueError(
                f"Profile {profile.profile_id} has {len(grades)} labels but there are "
                f"{len(opportunities)} opportunities ({_LABEL_ORDER_NOTE})."
            )
        for opportunity, grade in zip(opportunities, grades):
            mapping[(profile.profile_id, opportunity.id)] = int(grade)
    return mapping


def load() -> Dataset:
    """Build the v1 dataset: 8 listings, 4 short vague queries, hand-graded labels."""
    opportunities = [
        Opportunity(
            id=str(raw["id"]),
            title=str(raw["title"]),
            description=str(raw["description"]),
            skills=[str(s) for s in raw["required_skills"]],  # type: ignore[union-attr]
            eligibility=str(raw["eligibility"]),
        )
        for raw in _RAW_OPPORTUNITIES
    ]

    # For v1 the "query" is the student's own self-description only. Major and
    # year are carried as metadata and are NOT part of the searchable text.
    queries = [
        Query(
            profile_id=raw["profile_id"],
            text=raw["query"],
            major=raw["major"],
            year=raw["year"],
        )
        for raw in _RAW_PROFILES
    ]

    return Dataset(
        name="prototype",
        description=(
            "Original prototype: 8 listings, 4 short/vague profiles, "
            "hand-graded labels — these are the numbers recorded in the README."
        ),
        opportunities=opportunities,
        queries=queries,
        relevance_map=_build_relevance_map(opportunities, queries),
        suggested_top_k=5,
    )
