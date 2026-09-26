# import os
# import sys
# # 1. path to root folder
# project_root = os.path.dirname(os.path.abspath(__file__))
#
# # 2. Append it to sys.path so Python can find your local modules
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)


from rbiFaqAgent.RAG.MarkDown_chunking import extract_chunks_from_files
from Utils.config_loader import load_rag_config
from rbiFaqAgent.RAG.Vector_Audit import vector_audit
from rbiFaqAgent.RAG.vector_Store import save_documents_to_chroma

def main():
    try:
        # CHOOSE STRATEGY: Swap this string to alter your entire pipeline setup instantly
        ACTIVE_STRATEGY = "markdown_fixed_size"

        print(f"--- Booting Pipeline with Strategy: {ACTIVE_STRATEGY} ---")

        # 1. Load configuration parameters dynamically
        cfg = load_rag_config(strategy_name=ACTIVE_STRATEGY)

        # 2. Extract Document chunks using the loaded settings
        mydocs = extract_chunks_from_files(cfg=cfg)

        # 3. Store in Vector Database (Update vector_store.py functions to accept 'cfg' dict)
        save_documents_to_chroma(documents=mydocs, cfg=cfg, force_recreate=True)

        #4. Audit Vector DB
        vector_audit(cfg)

        print("--- Pipeline Execution Succeeded ---")

    except Exception as e:
        print(f"🛑 Pipeline Failed: {e}")

if __name__ == "__main__":
    print("--- Imports loaded successfully! ---")
    main()