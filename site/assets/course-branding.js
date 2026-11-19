/* Shared course mark, taken unchanged from the original lecture header. */
for(const section of document.querySelectorAll('.slides > section[data-part]')){
 if(!section.querySelector('.course-header-brand')){
  const image=document.createElement('img');image.className='course-header-brand';image.src='../assets/data-science-header.png';image.alt='Data Science Professional Certificate Program at LMU';section.appendChild(image);
 }
}
