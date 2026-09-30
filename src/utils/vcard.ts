export async function downloadVCard(photoUrl: string = '/assets/images/profile.jpg') {
  try {
    let base64Photo = '';
    try {
      const res = await fetch(photoUrl);
      const blob = await res.blob();
      base64Photo = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => {
          const result = reader.result as string;
          resolve(result.split(',')[1] || '');
        };
        reader.onerror = reject;
        reader.readAsDataURL(blob);
      });
    } catch (e) {
      console.warn('Could not encode vCard photo, generating text-only vCard', e);
    }

    const lines = [
      'BEGIN:VCARD',
      'VERSION:3.0',
      'N:Zambounis;Orestis;;;',
      'FN:Orestis Zambounis',
      'ORG:Zambounis Technology',
      'TITLE:Senior ML Engineer (Model Optimization) & Full-Stack Engineer',
      'EMAIL;TYPE=INTERNET;TYPE=WORK:info@orestis.ch',
      'EMAIL;TYPE=INTERNET;TYPE=HOME:me@orestisz.com',
      'TEL;TYPE=CELL:+41786373591',
      'ADR;TYPE=WORK:;;Av. Eugène-Rambert 30;Lausanne;;1005;Switzerland',
      'URL:https://orestis.ch/',
      'NOTE:Senior ML Engineer at Red Hat (Model Optimization) · ex Q-SYS/Seervision · ETH Zurich Alumnus'
    ];

    if (base64Photo) {
      lines.push('PHOTO;ENCODING=b;TYPE=JPEG:' + base64Photo);
    }

    lines.push('END:VCARD');

    const vcardContent = lines.join('\r\n');
    const blob = new Blob([vcardContent], { type: 'text/vcard;charset=utf-8' });
    const url = URL.createObjectURL(blob);

    const link = document.createElement('a');
    link.href = url;
    link.download = 'Orestis_Zambounis.vcf';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  } catch (err) {
    console.error('Failed to generate vCard:', err);
    alert('Unable to generate vCard file.');
  }
}
