#####################################################################################
# ----------------------------------- App launcher ----------------------------------
#####################################################################################

##########################
# ---- Depedendencies ----
##########################

import pathlib
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent.parent 
sys.path.append(str(PROJECT_ROOT))
from app.routers.predict_router import router as predict_router
from src.core.load_env import EnvLoader


################
# ---- main ----
################



app = FastAPI(title="Maize Disease API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router)