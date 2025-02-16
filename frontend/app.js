const API_URL = 'http://127.0.0.1:5000';

async function loadPrompts() {
    const response = await fetch('/prompts');
    const prompts = await response.json();

    const promptList = document.getElementById('prompt-list');
    promptList.innerHTML = '';

    Object.keys(prompts).forEach(role => {
        const div = document.createElement('div');
        div.textContent = role;
        div.onclick = () => {
            fetch(`/prompts/${role}`)
                .then(response => response.json())
                .then(data => {
                    document.getElementById('prompt-content').value = data[role];
                    document.getElementById('prompt-content').dataset.role = role;
                });
        };
        promptList.appendChild(div);
    });
}

async function savePrompt() {
    const role = document.getElementById('prompt-content').dataset.role;
    const content = document.getElementById('prompt-content').value;

    if (!role || !content) {
        alert('Please select a role and provide content before saving.');
        return;
    }

    await fetch('/prompts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ role: role, content: content })
    });

    loadPrompts();
}

loadPrompts();