#!/usr/bin/env python3
import io
import json
import os
import zipfile
import base64
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# API GitHub
GITHUB_API = "https://api.github.com"

# Chemins codés en dur
INPUT_CSV = "/Documents/Code_Smell_LLM/Code_Smell_LLM/Prevalence/Dataset/merged_repos.csv"
OUTPUT_DIR = "/Documents/Code_Smell_LLM/Code_Smell_LLM/Prevalence/Extraction_LLM_Files/Extract_LLM_Files"
MAX_WORKERS = 4

# Dossier pour déposer les repos complets
DOWNLOADED_REPOS_DIR = os.path.join(OUTPUT_DIR, "repos")


def make_github_session() -> requests.Session:
    session = requests.Session()
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        session.headers["Authorization"] = f"token {token}"
    session.headers["Accept"] = "application/vnd.github+json"
    session.headers["User-Agent"] = "llm-repo-scanner"
    return session


SESSION = make_github_session()


@dataclass
class LLMFile:
    owner: str
    repo: str
    ref: str
    path: str
    providers: Set[str] = field(default_factory=set)


@dataclass
class RepoResult:
    owner: str
    repo: str
    ref: Optional[str]
    files: List[LLMFile] = field(default_factory=list)
    error: Optional[str] = None
    zip_bytes: Optional[bytes] = None  # Nouveau: on garde le zip en mémoire


# On ne regarde plus que les .py
TEXT_EXTENSIONS = {
    ".py",
}

PROVIDER_PATTERNS: Dict[str, List[str]] = {
    "openai": [
        "openai",
        "gpt-4o",
        "gpt4o",
        "gpt-4",
        "gpt4",
        "gpt-3.5-turbo",
        "openai.chat.completions",
        "openai.responses",
    ],
    "anthropic": [
        "anthropic",
        "claude",
        "claude-",
        "ChatAnthropic",
    ],
    "gemini": [
        "google.generativeai",
        "genai",
        "gemini",
        "gemini-",
        "vertexai.generative_models",
    ],
    "mistral": [
        "mistral",
        "mistral-",
    ],
    "llama": [
        "llama",
        "llama3",
        "llama-3",
        "Meta Llama",
        "llama_index",
        "llamaindex",
        "LlamaIndex",
    ],
    "groq": [
        "groq",
        "Groq",
    ],
    "cohere": [
        "cohere",
        "Cohere",
    ],
    "ollama": [
        "ollama",
        "Ollama",
    ],
    "vllm": [
        "vllm",
        "VLLM",
    ],
    "litellm": [
        "litellm",
        "LiteLLM",
        "lite_llm",
    ],
    "langchain": [
        "langchain",
        "LangChain",
        "ChatOpenAI",
        "ChatGroq",
        "ChatVertexAI",
        "ChatGoogleGenerativeAI",
    ],
}


def has_text_extension(path: str) -> bool:
    lower = path.lower()
    for ext in TEXT_EXTENSIONS:
        if lower.endswith(ext):
            return True
    return False


def detect_providers(text: str) -> Set[str]:
    providers: Set[str] = set()
    lower = text.lower()
    for provider, patterns in PROVIDER_PATTERNS.items():
        for pattern in patterns:
            if pattern.lower() in lower:
                providers.add(provider)
                break
    return providers


def github_get_json(url: str, params: Optional[Dict] = None) -> Tuple[Optional[Dict], Optional[str]]:
    try:
        resp = SESSION.get(url, params=params, timeout=30)
    except Exception as exc:
        return None, f"request error {exc}"
    status = resp.status_code
    if status == 404:
        return None, "not found"
    if status >= 300:
        return None, f"http {status} for {url}"
    try:
        return resp.json(), None
    except Exception as exc:
        return None, f"json decode error {exc}"


def resolve_ref(owner: str, repo: str, commit_sha: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if isinstance(commit_sha, str) and commit_sha.strip():
        return commit_sha.strip(), None
    url = f"{GITHUB_API}/repos/{owner}/{repo}"
    data, err = github_get_json(url)
    if err is not None:
        return None, err
    if not isinstance(data, dict):
        return None, "unexpected repo metadata format"
    default_branch = data.get("default_branch")
    if not default_branch:
        return None, "no default_branch in repo metadata"
    return default_branch, None


def download_repo_zip(owner: str, repo: str, ref: str) -> Tuple[Optional[bytes], Optional[str]]:
    url = f"{GITHUB_API}/repos/{owner}/{repo}/zipball/{ref}"
    try:
        resp = SESSION.get(url, timeout=120)
    except Exception as exc:
        return None, f"zip request error {exc}"
    status = resp.status_code
    if status == 404:
        return None, "zip not found"
    if status >= 300:
        return None, f"http {status} for {url}"
    return resp.content, None


def scan_repo_from_zip(owner: str, repo: str, ref: str, zip_bytes: bytes) -> RepoResult:
    result = RepoResult(owner=owner, repo=repo, ref=ref, zip_bytes=zip_bytes)
    # Le zip contient un dossier racine avec un nom generique, on s en fiche
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            path_in_zip = info.filename
            if not has_text_extension(path_in_zip):
                continue
            try:
                with zf.open(info, "r") as f:
                    raw = f.read()
            except Exception:
                continue
            try:
                text = raw.decode("utf-8", errors="ignore")
            except Exception:
                continue
            providers = detect_providers(text)
            if providers:
                result.files.append(
                    LLMFile(
                        owner=owner,
                        repo=repo,
                        ref=ref,
                        path=path_in_zip,
                        providers=providers,
                    )
                )
    return result


def scan_repo_row(row: pd.Series) -> RepoResult:
    owner = str(row["owner"]).strip()
    repo = str(row["repo"]).strip()
    commit_sha = row.get("commit_sha")
    ref, err = resolve_ref(owner, repo, commit_sha if isinstance(commit_sha, str) else None)
    result = RepoResult(owner=owner, repo=repo, ref=ref)
    if err is not None or ref is None:
        result.error = f"failed to resolve ref {err}"
        return result
    zip_bytes, zerr = download_repo_zip(owner, repo, ref)
    if zerr is not None or zip_bytes is None:
        result.error = f"failed to download zip {zerr}"
        return result
    return scan_repo_from_zip(owner, repo, ref, zip_bytes)


def extract_full_repo(zip_bytes: bytes, target_dir: str) -> None:
    """
    Extrait le contenu complet du zip dans le dossier cible.
    GitHub met tout dans un sous-dossier avec un nom aléatoire, 
    on extrait tout et on supprime ce niveau de hiérarchie.
    """
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        # Trouver le dossier racine (premier niveau)
        root_folder = None
        for name in zf.namelist():
            if '/' in name:
                root_folder = name.split('/')[0]
                break
        
        # Extraire tous les fichiers
        for info in zf.infolist():
            # Enlever le dossier racine du chemin
            if root_folder and info.filename.startswith(root_folder + '/'):
                relative_path = info.filename[len(root_folder) + 1:]
            else:
                relative_path = info.filename
            
            if not relative_path:  # C'est le dossier racine lui-même
                continue
                
            target_path = os.path.join(target_dir, relative_path)
            
            if info.is_dir():
                os.makedirs(target_path, exist_ok=True)
            else:
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                try:
                    with zf.open(info) as source:
                        with open(target_path, 'wb') as target:
                            target.write(source.read())
                except Exception as e:
                    print(f"Warning: Could not extract {info.filename}: {e}")


def save_downloaded_repos(results: List[RepoResult], base_dir: str) -> None:
    """
    Sauvegarde tous les repos téléchargés dans leur intégralité.
    """
    saved_count = 0
    for res in results:
        if not res.zip_bytes or not res.ref:
            continue
        
        repo_dir = os.path.join(
            base_dir,
            f"{res.owner}__{res.repo}__{res.ref}",
        )
        
        # Créer le dossier pour ce repo
        os.makedirs(repo_dir, exist_ok=True)
        
        # Extraire le contenu complet du zip
        try:
            extract_full_repo(res.zip_bytes, repo_dir)
            saved_count += 1
            print(f"Extracted: {res.owner}/{res.repo} -> {repo_dir}")
        except Exception as e:
            print(f"Error extracting {res.owner}/{res.repo}: {e}")
    
    print(f"\nTotal repos extracted: {saved_count}/{len(results)}")


def run_scan(input_csv: str, out_dir: str, max_workers: int = 4) -> None:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(DOWNLOADED_REPOS_DIR, exist_ok=True)

    df = pd.read_csv(input_csv)
    required_cols = {"owner", "repo"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"input CSV must contain at least columns {required_cols}")
    if "commit_sha" not in df.columns:
        df["commit_sha"] = None

    results: List[RepoResult] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(scan_repo_row, row): idx
            for idx, row in df.iterrows()
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                res = future.result()
            except Exception as exc:
                row = df.iloc[idx]
                res = RepoResult(
                    owner=str(row["owner"]).strip(),
                    repo=str(row["repo"]).strip(),
                    ref=None,
                    files=[],
                    error=f"uncaught error {exc}",
                )
            results.append(res)

    llm_rows: List[Dict] = []
    repo_summary: Dict[str, Dict] = {}

    for res in results:
        key = f"{res.owner}/{res.repo}"
        files_info = []
        providers_repo: Set[str] = set()
        for f in res.files:
            prov_list = sorted(f.providers)
            llm_rows.append(
                {
                    "owner": f.owner,
                    "repo": f.repo,
                    "ref": f.ref,
                    "file_path": f.path,
                    "providers": ",".join(prov_list),
                }
            )
            files_info.append(
                {
                    "file_path": f.path,
                    "providers": prov_list,
                }
            )
            providers_repo.update(prov_list)
        repo_summary[key] = {
            "owner": res.owner,
            "repo": res.repo,
            "ref": res.ref,
            "num_llm_files": len(res.files),
            "providers": sorted(providers_repo),
            "files": files_info,
            "error": res.error,
        }

    llm_df = pd.DataFrame(llm_rows)
    llm_csv_path = os.path.join(out_dir, "llm_files.csv")
    llm_df.to_csv(llm_csv_path, index=False)

    summary_path = os.path.join(out_dir, "repo_summary.json")
    with open(summary_path, "w", encoding="utf8") as f:
        json.dump(repo_summary, f, indent=2, ensure_ascii=False)

    # Sauvegarder tous les repos téléchargés
    print("\nExtracting downloaded repositories...")
    save_downloaded_repos(results, DOWNLOADED_REPOS_DIR)

    print(f"\nSaved per file results to {llm_csv_path}")
    print(f"Saved per repo summary to {summary_path}")
    print(f"Downloaded repos saved to {DOWNLOADED_REPOS_DIR}")
    total_repos = len(results)
    with_llm = sum(1 for r in results if r.files)
    print(f"Scanned {total_repos} repositories, {with_llm} have at least one LLM file")


def main() -> None:
    run_scan(INPUT_CSV, OUTPUT_DIR, max_workers=MAX_WORKERS)


if __name__ == "__main__":
    main()