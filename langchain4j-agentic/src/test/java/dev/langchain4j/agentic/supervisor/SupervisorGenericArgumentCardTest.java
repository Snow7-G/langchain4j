package dev.langchain4j.agentic.supervisor;

import static org.assertj.core.api.Assertions.assertThat;

import dev.langchain4j.agentic.Agent;
import dev.langchain4j.agentic.AgenticServices;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.service.V;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;

/**
 * Regression test for how the supervisor renders argument types in its agent cards when the
 * declared Java type is generic.
 *
 * <p>An argument declared as {@code List<String>} carries a
 * {@link java.lang.reflect.ParameterizedType}. The card shown to the planning model must describe
 * that type, otherwise the model has to guess the shape of the argument. Note that this is about
 * the <em>rendered</em> card, not about the {@code AgentArgument} that is derived upstream: a test
 * that only asserts the derived argument would not catch a regression here.
 */
class SupervisorGenericArgumentCardTest {

    interface SearchSubAgent {

        @Agent("Searches records in the backing system")
        String search(@V("recordType") String recordType, @V("fields") List<String> fields);
    }

    /** Captures the first system message it is asked to process, then aborts the planning loop. */
    static class CapturingChatModel implements ChatModel {

        private final AtomicReference<String> systemMessage = new AtomicReference<>();

        @Override
        public ChatResponse doChat(ChatRequest chatRequest) {
            chatRequest.messages().stream()
                    .filter(SystemMessage.class::isInstance)
                    .findFirst()
                    .ifPresent(message -> systemMessage.compareAndSet(
                            null, ((SystemMessage) message).text()));
            throw new IllegalStateException("prompt captured, aborting the planning loop");
        }

        String capturedPrompt() {
            return systemMessage.get();
        }
    }

    @Test
    void agent_card_should_describe_generic_argument_types() {
        CapturingChatModel model = new CapturingChatModel();

        SearchSubAgent subAgent = AgenticServices.agentBuilder(SearchSubAgent.class)
                .chatModel(model)
                .build();

        SupervisorAgent supervisor = AgenticServices.supervisorBuilder()
                .chatModel(model)
                .subAgents(subAgent)
                .build();

        try {
            supervisor.invoke("find the open tickets");
        } catch (RuntimeException expected) {
            // The capturing model always throws; the prompt is all we need.
        }

        String card = model.capturedPrompt();
        assertThat(card).as("the planner prompt").isNotNull();

        // Scalars are already handled correctly today.
        assertThat(card).contains("recordType: String");

        // This is the regression: the generic type must survive into the rendered card.
        assertThat(card).contains("fields: List<String>");
    }
}
