# Minimal Personal Webpage [orestis-z.github.io](http://orestis-z.github.io/)

## Deploying to Github Pages

- Fork or Clone this repo
- Rename the repo to your desired github pages url (E.g yourname.github.io)
- Customize `_config.yml` as desired
- Push code to a `gh-pages` branch
- Visit your website at the github pages url you specified in the second step

Learn more about Github pages <a href="https://pages.github.com/">here</a>

## Usage

- Update `_config.yml` with your desired settings
- Update `_includes/about.html` with information about yourself

## Local Setup

- Install <a href="https://www.ruby-lang.org/en/">Ruby</a>, <a href="https://bundler.io/">Bundler</a>, <a href="https://jekyllrb.com/">Jekyll</a>, and <a href="https://nodejs.org/en/">NodeJS</a> `NodeJS`.
- Run `bundle install` from repo's root
- Run `bundle exec jekyll serve --config _config.yml,_config-dev.yml` to start the local server
- Vist website in your browser at `http://localhost:4000`
- Run link, image and script checks on the website with `bundle exec htmlproofer ./_site --only-4xx`

## License

MIT
