import json
import re
import argparse
import structlog
from typing import List, Generator, Optional, Dict, Any
from enum import Enum
from pathlib import Path
from pydantic_xml import BaseXmlModel, element
from pydantic import BaseModel, Field
import dotenv
import os
from rich import print
from rich.table import Table
from rich.console import Console

from vulnhuntr.symbol_finder import SymbolExtractor
from vulnhuntr.LLMs import Claude, ChatGPT, Ollama, Gemini
from vulnhuntr.prompts import *
from vulnhuntr.reporting import generate_sarif, generate_html_report

dotenv.load_dotenv()

structlog.configure(
    processors=[
        structlog.processors.JSONRenderer()
    ],
    logger_factory=structlog.WriteLoggerFactory(
        file=Path('vulnhuntr').with_suffix(".log").open("wt", encoding="utf-8")
    )
)

import faulthandler
faulthandler.enable()

log = structlog.get_logger("vulnhuntr")
console = Console()


class VulnType(str, Enum):
    LFI = "LFI"
    RCE = "RCE"
    SSRF = "SSRF"
    AFO = "AFO"
    SQLI = "SQLI"
    XSS = "XSS"
    IDOR = "IDOR"
    SSTI = "SSTI"
    DESERIALIZATION = "DESERIALIZATION"
    AUTH_BYPASS = "AUTH_BYPASS"
    PATH_TRAVERSAL = "PATH_TRAVERSAL"


class ContextCode(BaseModel):
    name: str = Field(description="Function or Class name")
    reason: str = Field(description="Brief reason why this function's code is needed for analysis")
    code_line: str = Field(description="The single line of code where this context object is referenced.")


class Response(BaseModel):
    scratchpad: str = Field(description="Your step-by-step analysis process. Output in plaintext with no line breaks.")
    analysis: str = Field(description="Your final analysis. Output in plaintext with no line breaks.")
    poc: str = Field(default="None", description="Proof-of-concept exploit, if applicable.")
    remediation: Optional[str] = Field(default="", description="Defensive remediation guidance explaining how to securely fix the issue.")
    confidence_score: int = Field(description="0-10, where 0 is no confidence and 10 is absolute certainty because you have the entire user input to server output code path.")
    vulnerability_types: List[VulnType] = Field(description="The types of identified vulnerabilities")
    context_code: List[ContextCode] = Field(default_factory=list, description="List of context code items requested for analysis, one function or class name per item. No standard library or third-party package code.")


class ReadmeContent(BaseXmlModel, tag="readme_content"):
    content: str


class ReadmeSummary(BaseXmlModel, tag="readme_summary"):
    readme_summary: str


class Instructions(BaseXmlModel, tag="instructions"):
    instructions: str


class ResponseFormat(BaseXmlModel, tag="response_format"):
    response_format: str


class AnalysisApproach(BaseXmlModel, tag="analysis_approach"):
    analysis_approach: str


class Guidelines(BaseXmlModel, tag="guidelines"):
    guidelines: str


class FileCode(BaseXmlModel, tag="file_code"):
    file_path: str = element()
    file_source: str = element()


class PreviousAnalysis(BaseXmlModel, tag="previous_analysis"):
    previous_analysis: str


class ExampleBypasses(BaseXmlModel, tag="example_bypasses"):
    example_bypasses: str


class CodeDefinition(BaseXmlModel, tag="code"):
    name: str = element()
    context_name_requested: str = element()
    file_path: str = element()
    source: str = element()


class CodeDefinitions(BaseXmlModel, tag="context_code"):
    definitions: List[CodeDefinition] = []


class RepoOps:
    def __init__(self, repo_path: Path | str) -> None:
        self.repo_path = Path(repo_path)
        self.to_exclude = {'/setup.py', '/test', '/example', '/docs', '/site-packages', '.venv', 'virtualenv', '/dist', '/build'}
        self.file_names_to_exclude = ['test_', 'conftest', '_test.py']

        patterns = [
            # Async
            r'async\s+def\s+\w+\(.*?request',
            # Gradio
            r'gr\.Interface\(.*?\)',
            r'gr\.Interface\.launch\(.*?\)',
            # Flask
            r'@app\.route\(.*?\)',
            r'@blueprint\.route\(.*?\)',
            r'class\s+\w+\(MethodView\):',
            r'@(?:app|blueprint)\.add_url_rule\(.*?\)',
            # FastAPI & Starlette
            r'@app\.(?:get|post|put|delete|patch|options|head|trace)\(.*?\)',
            r'@router\.(?:get|post|put|delete|patch|options|head|trace)\(.*?\)',
            r'Route\(.*?\)',
            # Django
            r'url\(.*?\)',
            r're_path\(.*?\)',
            r'path\(.*?\)',
            r'@channel_layer\.group_add',
            r'@database_sync_to_async',
            # Pyramid
            r'@view_config\(.*?\)',
            # Bottle
            r'@(?:route|get|post|put|delete|patch)\(.*?\)',
            # Tornado
            r'class\s+\w+\((?:RequestHandler|WebSocketHandler)\):',
            r'@tornado\.gen\.coroutine',
            r'@tornado\.web\.asynchronous',
            # WebSockets
            r'websockets\.serve\(.*?\)',
            r'@websocket\.(?:route|get|post|put|delete|patch|head|options)\(.*?\)',
            # aiohttp
            r'app\.router\.add_(?:get|post|put|delete|patch|head|options)\(.*?\)',
            r'@routes\.(?:get|post|put|delete|patch|head|options)\(.*?\)',
            # Sanic
            r'@app\.(?:route|get|post|put|delete|patch|head|options)\(.*?\)',
            r'@blueprint\.(?:route|get|post|put|delete|patch|head|options)\(.*?\)',
            # Falcon
            r'app\.add_route\(.*?\)',
            # CherryPy
            r'@cherrypy\.expose',
            # GraphQL
            r'class\s+\w+\(graphene\.ObjectType\):',
            r'@strawberry\.type',
            # Server startup code
            r'app\.run\(.*?\)',
            r'serve\(app,.*?\)',
            r'uvicorn\.run\(.*?\)',
            r'application\.listen\(.*?\)',
            r'run_server\(.*?\)',
            r'server\.start\(.*?\)',
            r'web\.run_app\(.*?\)',
        ]

        self.compiled_patterns = [re.compile(p) for p in patterns]

    def get_readme_content(self) -> Optional[str]:
        prioritized_patterns = ["[Rr][Ee][Aa][Dd][Mm][Ee].[Mm][Dd]", "[Rr][Ee][Aa][Dd][Mm][Ee].[Rr][Ss][Tt]"]
        for pattern in prioritized_patterns:
            for readme in self.repo_path.glob(pattern):
                try:
                    with readme.open(encoding='utf-8', errors='ignore') as f:
                        return f.read()
                except Exception:
                    pass

        for readme in self.repo_path.glob("[Rr][Ee][Aa][Dd][Mm][Ee]*.[Mm][DdRrSsTt]"):
            try:
                with readme.open(encoding='utf-8', errors='ignore') as f:
                    return f.read()
            except Exception:
                pass
        return None

    def get_relevant_py_files(self) -> List[Path]:
        files = []
        for f in self.repo_path.rglob("*.py"):
            f_str = str(f).replace('\\', '/').lower()
            if any(exclude in f_str for exclude in self.to_exclude):
                continue
            if any(fn in f.name for fn in self.file_names_to_exclude):
                continue
            files.append(f)
        return files

    def get_network_related_files(self, files: List[Path]) -> Generator[Path, None, None]:
        for py_f in files:
            try:
                with py_f.open(encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                if any(p.search(content) for p in self.compiled_patterns):
                    yield py_f
            except Exception:
                continue

    def get_files_to_analyze(self, analyze_path: Path | None = None) -> List[Path]:
        path_to_analyze = analyze_path or self.repo_path
        if path_to_analyze.is_file():
            return [path_to_analyze]
        elif path_to_analyze.is_dir():
            return list(path_to_analyze.rglob('*.py'))
        else:
            raise FileNotFoundError(f"Specified analyze path does not exist: {path_to_analyze}")


def extract_between_tags(tag: str, string: str, strip: bool = False) -> list[str]:
    ext_list = re.findall(f"<{tag}>(.+?)</{tag}>", string, re.DOTALL)
    if strip:
        ext_list = [e.strip() for e in ext_list]
    return ext_list


def initialize_llm(llm_arg: str, system_prompt: str = "", custom_model: Optional[str] = None) -> Claude | ChatGPT | Ollama | Gemini:
    llm_arg = llm_arg.lower()
    if llm_arg in ['gemini', 'google']:
        model_name = custom_model or os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
        llm = Gemini(model=model_name, system_prompt=system_prompt)
    elif llm_arg == 'claude':
        anth_model = custom_model or os.getenv("ANTHROPIC_MODEL", "claude-3-7-sonnet-latest")
        anth_base_url = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
        llm = Claude(anth_model, anth_base_url, system_prompt)
    elif llm_arg in ['gpt', 'openai']:
        openai_model = custom_model or os.getenv("OPENAI_MODEL", "gpt-4o")
        openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        llm = ChatGPT(openai_model, openai_base_url, system_prompt)
    elif llm_arg == 'ollama':
        ollama_model = custom_model or os.getenv("OLLAMA_MODEL", "llama3")
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/api/generate")
        llm = Ollama(ollama_model, ollama_base_url, system_prompt)
    else:
        raise ValueError(f"Invalid LLM argument: {llm_arg}\nValid options are: gemini, claude, gpt, ollama")
    return llm


def print_readable(report: Response) -> None:
    for attr, value in vars(report).items():
        print(f"[bold cyan]{attr}:[/bold cyan]")
        if isinstance(value, str):
            lines = value.split('\n')
            for line in lines:
                print(f"  {line}")
        elif isinstance(value, list):
            for item in value:
                print(f"  - {item}")
        else:
            print(f"  {value}")
        print('-' * 40)
    print()


def run():
    parser = argparse.ArgumentParser(
        description='VulnHuntr v2.0: AI-powered static code analysis tracing tainted user-input call chains.'
    )
    parser.add_argument('-r', '--root', type=str, required=True, help='Path to the root directory of the project')
    parser.add_argument('-a', '--analyze', type=str, help='Specific path or file within the project to analyze')
    parser.add_argument('-l', '--llm', type=str, choices=['gemini', 'claude', 'gpt', 'ollama'], default='claude', help='LLM provider to use (default: claude)')
    parser.add_argument('-m', '--model', type=str, help='Custom model identifier (e.g. gemini-2.5-pro, claude-3-7-sonnet-latest, gpt-4o)')
    parser.add_argument('-o', '--output', type=str, help='Output path to save report file')
    parser.add_argument('-f', '--format', type=str, choices=['terminal', 'json', 'sarif', 'html'], default='terminal', help='Output format (default: terminal)')
    parser.add_argument('--max-iterations', type=int, default=7, help='Maximum context retrieval iterations (default: 7)')
    parser.add_argument('-v', '--verbosity', action='count', default=0, help='Increase output verbosity (-v for INFO, -vv for DEBUG)')
    args = parser.parse_args()

    repo = RepoOps(args.root)
    code_extractor = SymbolExtractor(args.root)
    files = repo.get_relevant_py_files()

    if args.analyze:
        analyze_path = Path(args.analyze)
        if analyze_path.is_absolute():
            files_to_analyze = repo.get_files_to_analyze(analyze_path)
        else:
            files_to_analyze = repo.get_files_to_analyze(Path(args.root) / analyze_path)
    else:
        files_to_analyze = list(repo.get_network_related_files(files))

    files_to_analyze = list(files_to_analyze)
    print(f"[bold green]VulnHuntr v2.0[/bold green] | Root: {args.root} | Files identified: {len(files_to_analyze)} | Provider: {args.llm}")

    llm = initialize_llm(args.llm, custom_model=args.model)
    readme_content = repo.get_readme_content()

    if readme_content:
        log.info("Summarizing project README")
        summary_res = llm.chat(
            (ReadmeContent(content=readme_content[:15000]).to_xml() + b'\n' +
             Instructions(instructions=README_SUMMARY_PROMPT_TEMPLATE).to_xml()
            ).decode()
        )
        tags = extract_between_tags("summary", str(summary_res))
        summary = tags[0] if tags else str(summary_res)
        log.info("README summary complete", summary=summary)
    else:
        log.warning("No README found")
        summary = ''

    system_prompt = (
        Instructions(instructions=SYS_PROMPT_TEMPLATE).to_xml() + b'\n' +
        ReadmeSummary(readme_summary=summary).to_xml()
    ).decode()

    llm = initialize_llm(args.llm, system_prompt=system_prompt, custom_model=args.model)

    all_findings: List[Dict[str, Any]] = []

    for py_f in files_to_analyze:
        log.info("Performing initial analysis", file=str(py_f))
        try:
            with py_f.open(encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            log.warning(f"Could not read file {py_f}: {e}")
            continue

        if not len(content.strip()):
            continue

        print(f"\n[bold yellow]Auditing:[/bold yellow] {py_f}")
        print('-' * 50)

        user_prompt = (
            FileCode(file_path=str(py_f), file_source=content).to_xml() + b'\n' +
            Instructions(instructions=INITIAL_ANALYSIS_PROMPT_TEMPLATE).to_xml() + b'\n' +
            AnalysisApproach(analysis_approach=ANALYSIS_APPROACH_TEMPLATE).to_xml() + b'\n' +
            PreviousAnalysis(previous_analysis='').to_xml() + b'\n' +
            Guidelines(guidelines=GUIDELINES_TEMPLATE).to_xml() + b'\n' +
            ResponseFormat(response_format=json.dumps(Response.model_json_schema(), indent=4)).to_xml()
        ).decode()

        try:
            initial_analysis_report: Response = llm.chat(user_prompt, response_model=Response)
        except Exception as e:
            log.error(f"Error during initial analysis of {py_f}: {e}")
            continue

        log.info("Initial analysis complete", report=initial_analysis_report.model_dump())
        print_readable(initial_analysis_report)

        # Secondary iterative taint-chain analysis
        if initial_analysis_report.confidence_score > 0 and len(initial_analysis_report.vulnerability_types):
            for vuln_type in initial_analysis_report.vulnerability_types:
                vuln_key = vuln_type.value if hasattr(vuln_type, "value") else str(vuln_type)
                vuln_data = VULN_SPECIFIC_BYPASSES_AND_PROMPTS.get(vuln_key, {})
                stored_code_definitions = {}
                definitions = CodeDefinitions(definitions=[])
                same_context = False
                previous_analysis = ''
                previous_context_amount = 0
                secondary_analysis_report = initial_analysis_report

                for i in range(args.max_iterations):
                    log.info("Performing vuln-specific analysis", iteration=i, vuln_type=vuln_key, file=str(py_f))

                    if i > 0:
                        previous_context_amount = len(stored_code_definitions)
                        previous_analysis = secondary_analysis_report.analysis

                        for context_item in secondary_analysis_report.context_code:
                            if context_item.name not in stored_code_definitions:
                                match = code_extractor.extract(context_item.name, context_item.code_line, files)
                                if match:
                                    stored_code_definitions[context_item.name] = CodeDefinition(
                                        name=match['name'],
                                        context_name_requested=match['context_name_requested'],
                                        file_path=match['file_path'],
                                        source=match['source']
                                    )

                        definitions = CodeDefinitions(definitions=list(stored_code_definitions.values()))

                    bypasses_str = '\n'.join(vuln_data.get('bypasses', []))
                    prompt_str = vuln_data.get('prompt', f"Analyze the call chain for {vuln_key} vulnerabilities.")

                    vuln_specific_user_prompt = (
                        FileCode(file_path=str(py_f), file_source=content).to_xml() + b'\n' +
                        definitions.to_xml() + b'\n' +
                        ExampleBypasses(example_bypasses=bypasses_str).to_xml() + b'\n' +
                        Instructions(instructions=prompt_str).to_xml() + b'\n' +
                        AnalysisApproach(analysis_approach=ANALYSIS_APPROACH_TEMPLATE).to_xml() + b'\n' +
                        PreviousAnalysis(previous_analysis=previous_analysis).to_xml() + b'\n' +
                        Guidelines(guidelines=GUIDELINES_TEMPLATE).to_xml() + b'\n' +
                        ResponseFormat(response_format=json.dumps(Response.model_json_schema(), indent=4)).to_xml()
                    ).decode()

                    try:
                        secondary_analysis_report = llm.chat(vuln_specific_user_prompt, response_model=Response)
                    except Exception as e:
                        log.error(f"Error during secondary analysis: {e}")
                        break

                    log.info("Secondary analysis iteration finished", report=secondary_analysis_report.model_dump())

                    if args.verbosity > 0:
                        print_readable(secondary_analysis_report)

                    if not len(secondary_analysis_report.context_code):
                        break

                    if previous_context_amount >= len(stored_code_definitions) and i > 0:
                        if same_context:
                            break
                        same_context = True

                if secondary_analysis_report.confidence_score >= 5:
                    all_findings.append({
                        "file_path": str(py_f),
                        "report": secondary_analysis_report.model_dump()
                    })
                    if args.verbosity == 0:
                        print_readable(secondary_analysis_report)

    # Print summary table
    table = Table(title="VulnHuntr v2.0 Scan Summary")
    table.add_column("File", style="cyan")
    table.add_column("Vulnerability Types", style="magenta")
    table.add_column("Confidence", justify="right", style="green")
    table.add_column("PoC", style="yellow")

    for f in all_findings:
        rep = f["report"]
        table.add_row(
            f["file_path"],
            ", ".join([str(t) for t in rep.get("vulnerability_types", [])]),
            f"{rep.get('confidence_score', 0)}/10",
            rep.get("poc", "None")[:40] + ("..." if len(rep.get("poc", "None")) > 40 else "")
        )

    console.print(table)

    summary_stats = {
        "files_scanned": len(files_to_analyze),
        "total_findings": len(all_findings),
        "llm_usage": getattr(llm, "usage", None).summary() if hasattr(llm, "usage") else {}
    }

    # Handle Exports
    if args.output or args.format != 'terminal':
        fmt = args.format
        out_path = Path(args.output) if args.output else Path(f"vulnhuntr-report.{fmt if fmt != 'terminal' else 'json'}")

        if fmt == 'sarif':
            sarif_data = generate_sarif(all_findings, repo_root=args.root)
            out_path.write_text(json.dumps(sarif_data, indent=2), encoding='utf-8')
            print(f"[bold green]Exported SARIF report to:[/bold green] {out_path}")
        elif fmt == 'html':
            html_data = generate_html_report(all_findings, summary_stats)
            out_path.write_text(html_data, encoding='utf-8')
            print(f"[bold green]Exported HTML report to:[/bold green] {out_path}")
        elif fmt == 'json':
            out_path.write_text(json.dumps({"findings": all_findings, "stats": summary_stats}, indent=2), encoding='utf-8')
            print(f"[bold green]Exported JSON report to:[/bold green] {out_path}")


if __name__ == '__main__':
    run()
