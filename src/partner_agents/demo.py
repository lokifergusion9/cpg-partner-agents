"""The fictional sample brand pre-filled in demo mode (HLD §13, CLAUDE.md rule 12).

Everything here is written from scratch for an invented brand. It makes no claim about a real
company, product or certification, and nothing in it comes from a third-party script.
"""

from partner_agents.schemas import ApprovedFact, BrandProfile

_FACTS = [
    ("Our cookies are baked in small batches in a single bakery.", ["bakery", "quality"]),
    ("Every flavour is made without wheat, and we say so on the front of the pack.", ["diet"]),
    ("We sell in 6-count and 12-count boxes, with a shelf life of 9 months.", ["format", "shelf"]),
    ("We ship from one warehouse and can fill a first order within two weeks.", ["fulfilment"]),
    ("We are looking for partners who serve natural and specialty grocery shops.", ["channel"]),
    ("We can send sample boxes to a buyer who asks for them.", ["samples"]),
]

SAMPLE_BRAND = BrandProfile(
    brand_name="Hearthside Crumbs",
    product="wheat-free cookies",
    value_prop="small-batch wheat-free cookies for natural and specialty grocery",
    sender_name="Maya Rivera",
    sender_email="maya@hearthside-crumbs.example",
    founder_name="Maya Rivera",
    category="wheat-free cookies",
    target_regions=["California", "Oregon", "Washington"],
    timezone="America/Los_Angeles",
    sign_off="Thanks from all of us at Hearthside Crumbs",
    website=None,
    approved_facts=[
        ApprovedFact(id=f"F-{n}", text=text, source="brand_profile", tags=tags)
        for n, (text, tags) in enumerate(_FACTS, start=1)
    ],
)
