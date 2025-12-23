def render_opensearch_xml(*, base_url: str) -> str:
    return (
        f'<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<OpenSearchDescription xmlns="http://a9.com/-/spec/opensearch/1.1/">\n'
        f"  <ShortName>ROSE</ShortName>\n"
        f"  <Description>Search ROSE</Description>\n"
        f"  <InputEncoding>UTF-8</InputEncoding>\n"
        f'  <Url type="text/html" template="{base_url}/v1/search?q={{searchTerms}}"/>\n'
        f'  <Url type="application/json" template="{base_url}/v1/search?q={{searchTerms}}"/>\n'
        f"</OpenSearchDescription>\n"
    )
