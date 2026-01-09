from fastapi import APIRouter

from rose_server.routers.ask import router as ask_router
from rose_server.routers.exporter import router as exporter_router
from rose_server.routers.factsheets import router as factsheets_router
from rose_server.routers.importer import router as importer_router
from rose_server.routers.lenses import router as lenses_router
from rose_server.routers.messages import router as messages_router
from rose_server.routers.models import router as models_router
from rose_server.routers.nav import router as nav_router
from rose_server.routers.search import router as search_router
from rose_server.routers.threads import router as threads_router

router = APIRouter()
router.include_router(ask_router)
router.include_router(exporter_router)
router.include_router(factsheets_router)
router.include_router(importer_router)
router.include_router(lenses_router)
router.include_router(messages_router)
router.include_router(models_router)
router.include_router(nav_router)
router.include_router(search_router)
router.include_router(threads_router)
