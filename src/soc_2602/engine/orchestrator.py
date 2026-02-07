class ConversationOrchestrator:
    def step(self):
        # 1. Check if we need to switch topics
        if self.current_topic_exhausted():
            self.state.transition_counter = random.randint(3, 5)
            self.broadcast_system_instruction(
                f"STATUS: TRANSITIONING. TURNS REMAINING: {self.state.transition_counter}"
            )

        # 2. Generate response
        response = self.current_agent.generate()

        # 3. Decrement counter if in transition
        if self.state.transition_counter > 0:
            self.state.transition_counter -= 1
            if self.state.transition_counter == 0:
                self.databank.inject_new_topic()
