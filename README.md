# RAG app with Qdrant

## Running a Qdrant instance on docker in you locall machine

```python
docker run --name qdrant --restart unless-stopped -p 6333:6333 -v D:\Python\rag-qdrant:/qdrant/storage qdrant/qdrant
```

## Questions to ask our RAG system:

*1. How well the GPT4 performed on the professional and academic exams?*

*2. What are some of the limitations of GPT-4?*
