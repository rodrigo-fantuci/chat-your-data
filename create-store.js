import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { Document } from "langchain/document";
import { SupabaseVectorStore } from "langchain/vectorstores/supabase";
import { createClient } from "@supabase/supabase-js";
import * as dotenv from "dotenv";
dotenv.config();

async function loadDocs() {
    const loader = new PDFLoader("./data/rev_bichos.pdf", {
        splitPages: false,
    });
    let docs = await loader.load();

    // Removes empty lines and multiple spaces
    docs = docs.map((doc) => {
        doc.pageContent = doc.pageContent.replace(/\n/g, " ");
        doc.pageContent = doc.pageContent.replace(/\s+/g, " ");
        return doc;
    });

    return docs;
}

const splitDocs = async (docs) => {
    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 2000,
        chunkOverlap: 100,
    });

    const text = docs[0].pageContent;
    return await splitter.splitDocuments([new Document({ pageContent: text })]);
};

const main = async () => {
    const privateKey = process.env.SUPABASE_PRIVATE_KEY;
    if (!privateKey) throw new Error(`Variável não encontrada SUPABASE_PRIVATE_KEY`);

    const url = process.env.SUPABASE_URL;
    if (!url) throw new Error(`Variável não encontrada SUPABASE_URL`);

    const docs = await loadDocs();
    const splittedDocs = await splitDocs(docs);

    const embeddings = new OpenAIEmbeddings({
        openAIApiKey: process.env.OPENAI_KEY,
    });

    const client = createClient(url, privateKey);
    await SupabaseVectorStore.fromDocuments(splittedDocs, embeddings, {
        client,
        tableName: "documents",
        queryName: "match_documents",
    });
};

main();
