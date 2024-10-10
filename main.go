// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Command ragserver is an HTTP server that implements RAG (Retrieval
// Augmented Generation) using the Gemini model and Weaviate. See the
// accompanying README file for additional details.
package main

import (
	"cmp"
	"context"
	_ "embed"
	"fmt"
	"log"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"strings"

	ollama "github.com/ollama/ollama/api"
	"github.com/weaviate/weaviate-go-client/v4/weaviate"
	"github.com/weaviate/weaviate-go-client/v4/weaviate/graphql"
	"github.com/weaviate/weaviate/entities/models"
)

// This is a standard Go HTTP server. Server state is in the ragServer struct.
// The `main` function connects to the required services (Weaviate and Google
// AI), initializes the server state and registers HTTP handlers.
func main() {
	ctx := context.Background()
	wvClient, err := initWeaviate(ctx)
	if err != nil {
		log.Fatal(err)
	}

	ollamaHost, err := url.Parse("http://localhost:11434")
	if err != nil {
		log.Fatal(err)
	}

	ollamaC := ollama.NewClient(ollamaHost, http.DefaultClient)

	server := &ragServer{
		wvClient:     wvClient,
		ollamaClient: ollamaC,
	}

	mux := http.NewServeMux()
	mux.HandleFunc("GET /", server.getIndex)
	mux.HandleFunc("POST /add/", server.addDocumentsHandler)
	mux.HandleFunc("POST /query/", server.queryHandler)

	port := cmp.Or(os.Getenv("SERVERPORT"), "9020")
	address := "localhost:" + port
	log.Println("listening on", address)
	log.Fatal(http.ListenAndServe(address, mux))
}

type ragServer struct {
	wvClient     *weaviate.Client
	ollamaClient *ollama.Client
}

//go:embed static/index.html
var index string

func (rs *ragServer) getIndex(w http.ResponseWriter, req *http.Request) {
	fmt.Fprint(w, index)
}

func (rs *ragServer) addDocumentsHandler(w http.ResponseWriter, req *http.Request) {
	// Parse HTTP request from JSON.
	type document struct {
		Text string
	}
	type addRequest struct {
		Documents []document
	}
	ar := &addRequest{}

	err := readRequestJSON(req, ar)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Use the batch embedding API to embed all documents at once.
	batch := ollama.EmbedRequest{
		Model: "llama3.1",
	}
	var input []string
	for _, doc := range ar.Documents {
		input = append(input, doc.Text)
	}
	batch.Input = input
	log.Printf("invoking embedding model with %v documents", len(ar.Documents))
	rsp, err := rs.ollamaClient.Embed(req.Context(), &batch)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	if len(rsp.Embeddings) != len(ar.Documents) {
		http.Error(w, "embedded batch size mismatch", http.StatusInternalServerError)
		return
	}

	// Convert our documents - along with their embedding vectors - into types
	// used by the Weaviate client library.
	objects := make([]*models.Object, len(ar.Documents))
	for i, doc := range ar.Documents {
		objects[i] = &models.Object{
			Class: "Document",
			Properties: map[string]any{
				"text": doc.Text,
			},
			Vector: rsp.Embeddings[i],
		}
	}

	// Store documents with embeddings in the Weaviate DB.
	log.Printf("storing %v objects in weaviate", len(objects))
	_, err = rs.wvClient.Batch().ObjectsBatcher().WithObjects(objects...).Do(req.Context())
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
}

func toFloat32(fs []float64) []float32 {
	res := make([]float32, 0, len(fs))
	for _, f := range fs {
		res = append(res, float32(f))
	}
	return res
}

func (rs *ragServer) queryHandler(w http.ResponseWriter, req *http.Request) {
	// Parse HTTP request from JSON.
	type queryRequest struct {
		Content string
	}
	qr := &queryRequest{}
	err := readRequestJSON(req, qr)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	slog.Info("received prompt query", slog.String("query", qr.Content))

	// Embed the query contents.
	rsp, err := rs.ollamaClient.Embeddings(req.Context(), &ollama.EmbeddingRequest{
		Model:  "llama3.1",
		Prompt: qr.Content,
	})
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Search weaviate to find the most relevant (closest in vector space)
	// documents to the query.
	gql := rs.wvClient.GraphQL()
	result, err := gql.Get().
		WithNearVector(
			gql.NearVectorArgBuilder().WithVector(toFloat32(rsp.Embedding))).
		WithClassName("Document").
		WithFields(graphql.Field{Name: "text"}).
		WithLimit(3).
		Do(req.Context())
	if werr := combinedWeaviateError(result, err); werr != nil {
		http.Error(w, werr.Error(), http.StatusInternalServerError)
		return
	}

	contents, err := decodeGetResults(result)
	if err != nil {
		http.Error(w, fmt.Errorf("reading weaviate response: %w", err).Error(), http.StatusInternalServerError)
		return
	}

	slog.Info("retrieved context from weaviate", slog.String("context", strings.Join(contents, "\n")))

	// Create a RAG query for the LLM with the most relevant documents as
	// context.
	ragQuery := fmt.Sprintf(ragTemplateStr, qr.Content, strings.Join(contents, "\n"))

	useStreaming := false
	var resp ollama.GenerateResponse
	err = rs.ollamaClient.Generate(req.Context(), &ollama.GenerateRequest{
		Model:  "llama3.1",
		Prompt: ragQuery,
		Stream: &useStreaming,
	}, func(r ollama.GenerateResponse) error {
		resp = r
		return nil
	})
	if err != nil {
		log.Printf("calling generative model: %v", err.Error())
		http.Error(w, "generative model error", http.StatusInternalServerError)
		return
	}

	slog.Info("got response from llm", slog.String("response", resp.Response))

	renderJSON(w, struct {
		Response string `json:"response"`
	}{
		Response: resp.Response,
	})
}

const ragTemplateStr = `
I will ask you a question and will provide some additional context information.
Assume this context information is factual and correct, as part of internal
documentation.
If the question relates to the context, answer it using the context.
If the question does not relate to the context, answer it as normal.

For example, let's say the context has nothing in it about tropical flowers;
then if I ask you about tropical flowers, just answer what you know about them
without referring to the context.

For example, if the context does mention minerology and I ask you about that,
provide information from the context along with general knowledge.

Question:
%s

Context:
%s
`

// decodeGetResults decodes the result returned by Weaviate's GraphQL Get
// query; these are returned as a nested map[string]any (just like JSON
// unmarshaled into a map[string]any). We have to extract all document contents
// as a list of strings.
func decodeGetResults(result *models.GraphQLResponse) ([]string, error) {
	data, ok := result.Data["Get"]
	if !ok {
		return nil, fmt.Errorf("Get key not found in result")
	}
	doc, ok := data.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("Get key unexpected type")
	}
	slc, ok := doc["Document"].([]any)
	if !ok {
		return nil, fmt.Errorf("Document is not a list of results")
	}

	var out []string
	for _, s := range slc {
		smap, ok := s.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("invalid element in list of documents")
		}
		s, ok := smap["text"].(string)
		if !ok {
			return nil, fmt.Errorf("expected string in list of documents")
		}
		out = append(out, s)
	}
	return out, nil
}
