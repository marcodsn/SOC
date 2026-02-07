class DatabankManager:
    def __init__(self, full_experience_data):
        self.full_data = full_experience_data
        self.revealed_topics = []
        self.current_summary = "Chat just started."

    def update_summary(self, recent_history, llm_client):
        """Compresses old turns into a 'Tonal Summary'"""
        # Call LLM to summarize previous topic preserving emotion
        new_summary = llm_client.generate_summary(recent_history)
        self.current_summary = f"{self.current_summary}\n{new_summary}"

    def get_next_topic_inject(self):
        """Fetches the next topic to 'reveal' to the agents"""
        # Logic to drip-feed information
        pass
