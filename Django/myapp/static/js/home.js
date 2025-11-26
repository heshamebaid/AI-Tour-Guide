console.log('home.js loaded');
const text = 'AI-Powered Tour Guide Platform for Ancient Egyptian History';
const target = document.getElementById('home-title');
let index = 0;

function typeLetter() {
  if (!target) return;
  if (index < text.length) {
    target.innerHTML += text[index] === '\n' ? '<br>' : text[index];
    index++;
    setTimeout(typeLetter, 40); 
  }
}

if (target) typeLetter();