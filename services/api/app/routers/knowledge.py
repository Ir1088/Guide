from fastapi import APIRouter, UploadFile

from app.services.insight_service import list_knowledge_documents
from app.services.rag_eval_service import evaluate_rag_accuracy

router = APIRouter()


@router.post("/documents")
async def upload_document(file: UploadFile) -> dict[str, object]:
    return {
        "filename": file.filename or "unknown",
        "status": "uploaded",
        "statusLabel": "已上传，等待解析",
        "next_step": "parse_chunk_embed_index",
        "document": {
            "id": "uploaded-preview",
            "name": file.filename or "unknown",
            "type": file.content_type or "application/octet-stream",
            "status": "uploaded",
            "statusLabel": "已上传",
            "updatedAt": "-",
            "chunkCount": 0,
            "source": "runtime_upload",
            "owner": "管理后台主线",
        },
    }


@router.get("/documents")
def list_documents() -> dict[str, object]:
    return list_knowledge_documents()


@router.get("/evaluate")
async def evaluate_knowledge(
    limit: int | None = None,
    use_reranker: bool = False,
    require_pgvector: bool = False,
) -> dict[str, object]:
    return await evaluate_rag_accuracy(
        limit,
        use_reranker=use_reranker,
        require_pgvector=require_pgvector,
    )
