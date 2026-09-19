LFI_TEMPLATE = """
Combine the code in <file_code> and <context_code> then analyze the code for remotely-exploitable Local File Inclusion (LFI) vulnerabilities by following the remote user-input call chain of code.

LFI-Specific Focus Areas:
1. High-Risk Functions and Methods:
   - open(), file(), io.open()
   - os.path.join() for file paths
   - Custom file reading functions

2. Path Traversal Opportunities:
   - User-controlled file paths or names
   - Dynamic inclusion of files or modules

3. File Operation Wrappers:
   - Template engines with file inclusion features
   - Custom file management classes

4. Indirect File Inclusion:
   - Configuration file parsing
   - Plugin or extension loading systems
   - Log file viewers

5. Example LFI-Specific Bypass Techniques are provided in <example_bypasses></example_bypasses> tags

When analyzing, consider:
- How user input influences file paths or names
- Effectiveness of path sanitization and validation
- Potential for null byte injection or encoding tricks
- Interaction with file system access controls
"""

RCE_TEMPLATE = """
Combine the code in <file_code> and <context_code> tags then analyze for remotely-exploitable Remote Code Execution (RCE) vulnerabilities by following the remote user-input call chain of code.

RCE-Specific Focus Areas:
1. High-Risk Functions and Methods:
   - eval(), exec(), subprocess modules (Popen, call, check_output)
   - os.system(), os.popen(), posix_spawn
   - pickle.loads(), yaml.load() (without SafeLoader), jsonpickle

2. Indirect Code Execution:
   - Dynamic imports (e.g., __import__(), importlib.import_module)
   - Reflection/introspection misuse (getattr, setattr)
   - Server-side template injection leading to code execution

3. Command Injection Vectors:
   - Shell command composition (shell=True in subprocess)
   - Unsanitized string concatenation or f-strings in system calls

4. Deserialization Vulnerabilities:
   - Unsafe deserialization of user-controlled data

5. Example RCE-Specific Bypass Techniques are provided in <example_bypasses></example_bypasses> tags.

When analyzing, consider:
- How user input flows into these high-risk areas
- Potential for filter evasion or sanitization bypasses
- Environment-specific factors affecting exploitability
"""

XSS_TEMPLATE = """
Combine the code in <file_code> and <context_code> tags then analyze for remotely-exploitable Cross-Site Scripting (XSS) vulnerabilities by following the remote user-input call chain of code.

XSS-Specific Focus Areas:
1. High-Risk Functions and Methods:
   - HTML rendering functions without autoescaping (markupsafe.Markup, flask.render_template_string)
   - Dynamic HTML response generation (Response(content, mimetype='text/html'))
   - DOM manipulation scripts or template reflection

2. Output Contexts:
   - Unescaped output in HTML body or attribute contexts
   - Direct injection into <script> tags or JavaScript event handlers

3. Input Handling:
   - User input reflection in responses without contextual encoding
   - Flawed custom regex sanitizers or tag strippers

4. Example XSS-Specific Bypass Techniques are provided in <example_bypasses></example_bypasses> tags.
"""

AFO_TEMPLATE = """
Combine the code in <file_code> and <context_code> tags then analyze for remotely-exploitable Arbitrary File Overwrite (AFO) vulnerabilities by following the remote user-input call chain of code.

AFO-Specific Focus Areas:
1. High-Risk Functions and Methods:
   - open() with write/append modes ('w', 'wb', 'a')
   - os.rename(), shutil.move(), shutil.copy(), shutil.copyfile()
   - tarfile.extractall(), zipfile.ZipFile.extractall() without path validation (Zip Slip)

2. Path Traversal in File Destination:
   - User-controlled filenames or directory paths
   - Symlink creation or overwrite

3. Indirect File Writes:
   - Critical system configuration file overwrites
   - Cache file or Python source file overwrites enabling secondary code execution

4. Example AFO-Specific Bypass Techniques are provided in <example_bypasses></example_bypasses> tags.
"""

SSRF_TEMPLATE = """
Combine the code in <file_code> and <context_code> tags then analyze for remotely-exploitable Server-Side Request Forgery (SSRF) vulnerabilities by following the remote user-input call chain of code.

SSRF-Specific Focus Areas:
1. High-Risk Functions and Methods:
   - requests.get/post, httpx.get/post, aiohttp.ClientSession
   - urllib.request.urlopen, urllib2, http.client
   - Webhook triggers, proxy endpoints, remote file downloaders

2. URL Parsing Flaws & Bypasses:
   - Incomplete URL validation (failure to block 127.0.0.1, 0.0.0.0, 169.254.169.254)
   - DNS Rebinding or decimal/hex IP encoding bypasses
   - Alternative protocols (file://, dict://, gopher://)

3. Cloud Metadata Access:
   - Requests targeting AWS (169.254.169.254), GCP (metadata.google.internal), or Kubernetes APIs

4. Example SSRF-Specific Bypass Techniques are provided in <example_bypasses></example_bypasses> tags.
"""

SQLI_TEMPLATE = """
Combine the code in <file_code> and <context_code> tags then analyze for remotely-exploitable SQL Injection (SQLI) vulnerabilities by following these steps:

1. Identify Entry Points:
   - Locate points where remote user input is received.

2. Trace Input Flow:
   - Follow input into database query construction.

3. Locate SQL Operations:
   - Direct SQL query construction (cursor.execute with string formatting/concatenation)
   - ORM methods accepting raw SQL (Model.objects.raw(), session.execute(text(...)), extra())
   - Dynamic column names, table names, or ORDER BY / GROUP BY clauses that bypass parameterization

4. Evaluate Security Controls & Bypasses:
   - Test for parameterization gaps, second-order SQL injection, or flawed escaping.

5. Example SQLI-Specific Bypass Techniques are provided in <example_bypasses></example_bypasses> tags.
"""

IDOR_TEMPLATE = """
Combine the code in <file_code> and <context_code> tags then analyze for remotely-exploitable Insecure Direct Object Reference (IDOR) / Broken Object Level Authorization vulnerabilities.

IDOR-Specific Focus Areas:
1. Identifier Usage:
   - Check if IDs (user_id, document_id, tenant_id) are accepted directly from route params, query args, or request bodies.

2. Ownership & Authorization Verification:
   - Check whether the query fetches the object strictly filtering by the authenticated session owner (e.g., `WHERE id = ? AND user_id = current_user.id`), or blindly trusts the supplied ID.
   - Look for missing authorization checks before update, delete, or read operations.

3. Example IDOR-Specific Bypass Techniques are provided in <example_bypasses></example_bypasses> tags.
"""

SSTI_TEMPLATE = """
Combine the code in <file_code> and <context_code> tags then analyze for remotely-exploitable Server-Side Template Injection (SSTI) vulnerabilities.

SSTI-Specific Focus Areas:
1. High-Risk Functions and Methods:
   - Jinja2: jinja2.Template(user_input).render(), render_template_string(user_input)
   - Mako: mako.template.Template(user_input).render()
   - Tornado: tornado.template.Template(user_input).generate()
   - Django: Template(user_input).render(Context(...))

2. Flaws in Template Construction:
   - User input formatted or concatenated into the template string rather than passed as context parameters.
   - Template engine sandbox escapes via Python object traversal (__class__, __base__, __subclasses__).

3. Example SSTI-Specific Bypass Techniques are provided in <example_bypasses></example_bypasses> tags.
"""

DESERIALIZATION_TEMPLATE = """
Combine the code in <file_code> and <context_code> tags then analyze for remotely-exploitable Insecure Deserialization vulnerabilities.

Deserialization-Specific Focus Areas:
1. High-Risk Libraries & Functions:
   - pickle.loads(), pickle.load(), _pickle.loads()
   - yaml.load(data, Loader=yaml.Loader) or yaml.unsafe_load()
   - jsonpickle.decode(), marshal.loads(), shelve.open()
   - torch.load() without weights_only=True

2. Source of Serialized Data:
   - Deserializing user-supplied cookies, tokens, request body data, or uploaded files.
   - Base64 encoded or hex encoded blobs decoded and deserialized.

3. Example Deserialization-Specific Bypass Techniques are provided in <example_bypasses></example_bypasses> tags.
"""

AUTH_BYPASS_TEMPLATE = """
Combine the code in <file_code> and <context_code> tags then analyze for remotely-exploitable Broken Authentication & Authorization Bypasses.

Auth Bypass Focus Areas:
1. Missing Authentication Decorators / Middleware:
   - Endpoints exposing sensitive actions (admin actions, data export, role assignment) without authentication dependencies.

2. Flawed Token / Session Verification:
   - JWT validation with algorithm "none", missing signature verification, or hardcoded secrets.
   - Insecure password reset or account recovery tokens.
   - Header spoofing (trusting X-Forwarded-For, X-Original-URL, or custom auth headers without proxy validation).

3. Example Auth Bypass Techniques are provided in <example_bypasses></example_bypasses> tags.
"""

PATH_TRAVERSAL_TEMPLATE = """
Combine the code in <file_code> and <context_code> tags then analyze for remotely-exploitable Path Traversal and Zip Slip vulnerabilities.

Path Traversal Focus Areas:
1. File Path Resolution:
   - Concatenating user input with base directories without verifying that the resolved canonical path starts with the intended base directory.
   - Flawed validation using `startswith` on non-canonicalized paths or stripping only single `../`.

2. Archive Extraction (Zip Slip):
   - Using `extractall()` on tar/zip archives without verifying each member's filename for `../` or leading `/`.

3. Example Path Traversal Techniques are provided in <example_bypasses></example_bypasses> tags.
"""

VULN_SPECIFIC_BYPASSES_AND_PROMPTS = {
    "LFI": {
        "prompt": LFI_TEMPLATE,
        "bypasses": [
            "../../../../etc/passwd",
            "/proc/self/environ",
            "data://text/plain;base64,PD9waHAgc3lzdGVtKCRfR0VUWydjbWQnXSk7Pz4=",
            "file:///etc/passwd",
            "C:\\win.ini",
            "/?../../../../../../../etc/passwd",
            "..%252f..%252f..%252fetc%252fpasswd"
        ]
    },
    "RCE": {
        "prompt": RCE_TEMPLATE,
        "bypasses": [
            "__import__('os').system('id')",
            "eval('__import__(\\'os\\').popen(\\'id\\').read()')",
            "exec('import subprocess;print(subprocess.check_output([\\'id\\']))')",
            "globals()['__builtins__'].__import__('os').system('id')",
            "getattr(__import__('os'), 'system')('id')",
            "$(touch${IFS}/tmp/mcinerney)",
            "; id;",
            "import pickle; pickle.loads(b'cos\\nsystem\\n(S\"id\"\\ntR.')"
        ]
    },
    "SSRF": {
        "prompt": SSRF_TEMPLATE,
        "bypasses": [
            "http://0.0.0.0:22",
            "file:///etc/passwd",
            "http://127.0.0.1:80",
            "http://169.254.169.254/latest/meta-data/",
            "http://[::1]/",
            "dict://127.0.0.1:11211/",
            "gopher://127.0.0.1:9000/_GET /"
        ]
    },
    "AFO": {
        "prompt": AFO_TEMPLATE,
        "bypasses": [
            "../../../etc/passwd%00.jpg",
            "shell.py;.jpg",
            ".htaccess",
            "/proc/self/cmdline",
            "../../config.py/."
        ]
    },
    "SQLI": {
        "prompt": SQLI_TEMPLATE,
        "bypasses": [
            "' UNION SELECT username, password FROM users--",
            "1 OR 1=1--",
            "admin'--",
            "1; DROP TABLE users--",
            "' OR '1'='1"
        ]
    },
    "XSS": {
        "prompt": XSS_TEMPLATE,
        "bypasses": [
            "<script>alert(document.domain)</script>",
            "\" onfocus=\"alert(1)\" autofocus=\"",
            "javascript:alert(1)",
            "'-alert(1)-'",
            "<img src=x onerror=alert(1)>"
        ]
    },
    "IDOR": {
        "prompt": IDOR_TEMPLATE,
        "bypasses": [
            "Replacing resource ID 123 with target user ID 124",
            "Parameter pollution: id=123&id=124",
            "JSON body tampering: {\"id\": 124, \"role\": \"admin\"}"
        ]
    },
    "SSTI": {
        "prompt": SSTI_TEMPLATE,
        "bypasses": [
            "{{7*7}}",
            "{{config}}",
            "{{self._TemplateReference__context.cycler.__init__.__globals__.os.popen('id').read()}}",
            "{{request.application.__globals__.__builtins__.__import__('os').popen('id').read()}}",
            "${''.getClass().forName('java.lang.Runtime').getMethod('getRuntime',null).invoke(null,null).exec('id')}"
        ]
    },
    "DESERIALIZATION": {
        "prompt": DESERIALIZATION_TEMPLATE,
        "bypasses": [
            "b'cos\\nsystem\\n(S\"id\"\\ntR.'",
            "!!python/object/apply:os.system [\"id\"]",
            "{\"py/object\": \"os.system\", \"py/tuple\": [\"id\"]}"
        ]
    },
    "AUTH_BYPASS": {
        "prompt": AUTH_BYPASS_TEMPLATE,
        "bypasses": [
            "Manipulating JWT alg to 'none'",
            "Accessing unauthenticated internal endpoint /api/v1/internal/admin",
            "Spoofing X-Forwarded-For: 127.0.0.1 or X-Remote-User: admin"
        ]
    },
    "PATH_TRAVERSAL": {
        "prompt": PATH_TRAVERSAL_TEMPLATE,
        "bypasses": [
            "../../../../etc/passwd",
            "..%2f..%2f..%2fetc/passwd",
            "....//....//....//etc/passwd",
            "/var/www/uploads/../../../../etc/passwd"
        ]
    }
}

INITIAL_ANALYSIS_PROMPT_TEMPLATE = """
Analyze the code in <file_code> tags for potential remotely exploitable vulnerabilities:
1. Identify all remote user input entry points (e.g., API endpoints, route handlers, form submissions, WebSocket handlers) and if you cannot find that, request the necessary classes or functions in the <context_code> tags.
2. Locate potential vulnerability sinks for:
   - Local File Inclusion (LFI)
   - Arbitrary File Overwrite (AFO)
   - Server-Side Request Forgery (SSRF)
   - Remote Code Execution (RCE)
   - Cross-Site Scripting (XSS)
   - SQL Injection (SQLI)
   - Insecure Direct Object Reference (IDOR)
   - Server-Side Template Injection (SSTI)
   - Insecure Deserialization (DESERIALIZATION)
   - Broken Authentication / Authorization (AUTH_BYPASS)
   - Path Traversal / Zip Slip (PATH_TRAVERSAL)
3. Note any security controls, validation, or sanitization measures encountered along the way so you can assess bypass possibilities.
4. Highlight areas where more context is needed to complete the analysis.
5. Provide actionable defensive remediation advice explaining how to securely fix the issue.

Be generous and thorough in identifying potential vulnerabilities as you will analyze more code in subsequent steps. If there is a plausible vulnerability, include it in the vulnerability_types field.
"""

README_SUMMARY_PROMPT_TEMPLATE = """
Provide a very concise summary of the README content in <readme_content></readme_content> tags from a security researcher's perspective, focusing specifically on:
1. The project's main purpose
2. Any networking capabilities, such as web interfaces, API endpoints, microservices, or remote network protocols
3. Key features that handle external untrusted data or network communications

Please keep the summary brief and to the point, highlighting only the most relevant functionality as it relates to attack surface.
Output in <summary></summary> XML tags.
"""

GUIDELINES_TEMPLATE = """Reporting Guidelines:
1. JSON Format:
   - Provide a single, well-formed JSON report combining all findings.
   - Place your step-by-step analysis in the scratchpad field before doing a final analysis in the analysis field.
   - Include clear defensive remediation recommendations in the remediation field.

2. Context Requests:
   - Classes: Use ClassName
   - Functions: Use func_name or ClassName.method_name
   - If you request ClassName, do not also request ClassName.method_name as that code will already be fetched with the ClassName request.
   - Do not request code from standard libraries or third-party packages; rely on standard specifications.

3. Vulnerability Reporting:
   - Report only remotely exploitable vulnerabilities (not purely local CLI tool arguments).
   - Provide a confidence score (0-10) and detailed justification for each vulnerability:
     - 0-4: Low likelihood / missing user-input confirmation
     - 5-7: Moderate / suspicious pattern requiring further context
     - 8-10: High / full verified path from untrusted remote input to dangerous sink.
   
4. Proof of Concept:
   - Include a PoC exploit or detailed exploitation steps specific to the analyzed codebase.
   - Ensure the PoC demonstrates how existing sanitization or filters are bypassed.
"""

ANALYSIS_APPROACH_TEMPLATE = """Analysis Instructions:
1. Comprehensive Review:
   - Thoroughly examine the content in <file_code> and <context_code> tags with a focus on remotely exploitable vulnerabilities.

2. Vulnerability Scanning:
   - Identify entry points for remote user input and trace them to sensitive functions/sinks.
   - Consider non-obvious attack vectors, chained logic flaws, and encoding edge cases.

3. Code Path Analysis:
   - Trace the flow of user input from remote request source to function sink.
   - Examine input validation, sanitization, and type coercions.

4. Security Control Analysis:
   - Evaluate each security measure's effectiveness against known bypass techniques.

5. Context-Aware Iteration:
   - Build upon previous findings in <previous_analysis> using new definitions provided in <context_code>.
   - Request additional context code as needed to verify the call chain.

6. Remediation Guidance:
   - Provide specific defensive engineering guidance to patch any confirmed vulnerability.
"""

SYS_PROMPT_TEMPLATE = """
You are the world's foremost expert in software security analysis, renowned for uncovering novel vulnerabilities and logic flaws in web applications and backend systems. Your task is to perform an exhaustive static code analysis, focusing on remotely exploitable vulnerabilities including:

1. Local File Inclusion (LFI)
2. Remote Code Execution (RCE)
3. Server-Side Request Forgery (SSRF)
4. Arbitrary File Overwrite (AFO)
5. SQL Injection (SQLI)
6. Cross-Site Scripting (XSS)
7. Insecure Direct Object References (IDOR)
8. Server-Side Template Injection (SSTI)
9. Insecure Deserialization (DESERIALIZATION)
10. Broken Authentication / Authorization (AUTH_BYPASS)
11. Path Traversal & Zip Slip (PATH_TRAVERSAL)

Your analysis must:
- Meticulously track user input from remote sources to high-risk function sinks.
- Uncover complex, multi-step vulnerabilities that may bypass security controls.
- Provide actionable, defensive remediation guidance for each identified issue.
- Strategically request necessary context code in the context_code field if call chains are incomplete.

The project's README summary is provided in <readme_summary> tags to help understand the attack surface.
Output your findings in JSON format conforming to the requested schema.
"""