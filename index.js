import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { OpenAI } from "langchain/llms/openai";
import { ConversationalRetrievalQAChain } from "langchain/chains";
import { SupabaseVectorStore } from "langchain/vectorstores/supabase";
import { createClient } from "@supabase/supabase-js";
import readline from "readline";
import * as dotenv from "dotenv";
dotenv.config();

try {
    const privateKey = process.env.SUPABASE_PRIVATE_KEY;
    const url = process.env.SUPABASE_URL;

    const client = createClient(url, privateKey);

    const vectorStore = new SupabaseVectorStore(
        new OpenAIEmbeddings({
            openAIApiKey: process.env.OPENAI_KEY,
        }),
        {
            client,
        }
    );

    let rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
    });

    const model = new OpenAI({
        openAIApiKey: process.env.OPENAI_KEY,
    });

    const chain = ConversationalRetrievalQAChain.fromLLM(model, vectorStore.asRetriever());
    let chatHistory = [];

    function getPrompt() {
        rl.question("Prompt: ", async (prompt) => {
            const res = await chain.call({ question: prompt, chat_history: chatHistory });
            console.log("Resposta: ", res.text);
            chatHistory.push(prompt + res.text);
            getPrompt();
        });
    }

    getPrompt();
} catch (e) {
    console.log(e);
}
