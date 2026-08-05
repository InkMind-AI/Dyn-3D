function copyBibTeX() {
    const bibtexElement = document.getElementById('bibtex-code');
    const label = document.getElementById('copy-label');
    if (!bibtexElement || !label) return;

    const text = bibtexElement.textContent;
    const showCopied = () => {
        label.textContent = 'Copied!';
        setTimeout(() => { label.textContent = 'Copy'; }, 2000);
    };

    navigator.clipboard.writeText(text).then(showCopied).catch(function() {
        const textArea = document.createElement('textarea');
        textArea.value = text;
        document.body.appendChild(textArea);
        textArea.select();
        document.execCommand('copy');
        document.body.removeChild(textArea);
        showCopied();
    });
}
