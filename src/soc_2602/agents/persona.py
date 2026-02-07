from dataclasses import dataclass


@dataclass
class PersonaState:
    permanent: str  # "I am a 35yo Accountant..."
    instant: str  # "I just burned my tongue on hot coffee."

    def get_system_block(self) -> str:
        # Renders the natural language block for the prompt
        return f"""
        [WHO YOU ARE]
        {self.permanent}

        [CURRENT STATE]
        {self.instant}
        """
