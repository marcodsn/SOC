"""
Generation modules for the SOC pipeline.

This subpackage contains the three core generators that form the
SOC data-creation pipeline:

1. :class:`PersonaGenerator` — generates diverse, psychologically
   grounded character personas from demographic seeds and few-shot
   examples.

2. :class:`ExperienceGenerator` — pairs two personas and synthesizes
   a realistic conversational context (relationship, situation, topic
   roadmap, and instant events).

3. :class:`ConversationGenerator` — runs a turn-by-turn conversation
   loop between two persona agents grounded in a shared experience,
   with rolling summarization and topic-state tracking.
"""

from soc_2602.generation.conversation_generator import ConversationGenerator
from soc_2602.generation.experience_generator import ExperienceGenerator
from soc_2602.generation.persona_generator import PersonaGenerator

__all__ = [
    "PersonaGenerator",
    "ExperienceGenerator",
    "ConversationGenerator",
]
