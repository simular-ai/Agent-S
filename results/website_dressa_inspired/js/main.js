// Minimal catalog, cart and modal logic
const PRODUCTS = [
  {id:1,title:'Літня сукня',price:899,image:'https://images.unsplash.com/photo-1520975925553-66f1d6a9c4b1?w=800&q=80'},
  {id:2,title:'Тренч класичний',price:1299,image:'https://images.unsplash.com/photo-1495121605193-b116b5b09d0b?w=800&q=80'},
  {id:3,title:'Джинси',price:649,image:'https://images.unsplash.com/photo-1514996937319-344454492b37?w=800&q=80'},
  {id:4,title:'В'язаний кардиган',price:499,image:'https://images.unsplash.com/photo-1541099649105-f69ad21f3246?w=800&q=80'}
];

function renderCatalog(filter=''){
  const container = document.getElementById('catalog');
  container.innerHTML = '';
  const f = filter.trim().toLowerCase();
  PRODUCTS.filter(p=>p.title.toLowerCase().includes(f)).forEach(p=>{
    const el = document.createElement('div'); el.className='card';
    el.innerHTML = `<img src="${p.image}" alt="${p.title}"><h4>${p.title}</h4><p>${p.price} ₴</p><p><a href="product.html?id=${p.id}" class="btn">Деталі</a> <button class="btn" onclick="addCart(${p.id})">Додати</button></p>`;
    container.appendChild(el);
  })
}
function addCart(id){
  const cart = JSON.parse(localStorage.getItem('cart')||'{}');
  cart[id] = (cart[id]||0)+1; localStorage.setItem('cart',JSON.stringify(cart));
  alert('Додано до кошика'); updateCartCount();
}
function updateCartCount(){
  const cart = JSON.parse(localStorage.getItem('cart')||'{}');
  const count = Object.values(cart).reduce((a,b)=>a+b,0);
  const el = document.getElementById('cartCount'); if(el) el.textContent = count;
}
function init(){
  const search = document.getElementById('search'); if(search){ search.addEventListener('input', e=>renderCatalog(e.target.value)); }
  updateCartCount(); renderCatalog();
  const resBtn = document.getElementById('reserveBtn');
  const modal = document.getElementById('reserveModal');
  const close = document.getElementById('closeModal');
  resBtn && resBtn.addEventListener('click', e=>{ e.preventDefault(); modal.classList.add('open'); });
  close && close.addEventListener('click', ()=>modal.classList.remove('open'));
}
window.addEventListener('DOMContentLoaded', init);