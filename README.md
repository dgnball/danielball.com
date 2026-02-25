# danielball.com

The source code for [danielball.com](http://danielball.com), my personal portfolio and notes site.

## About

This site started life a long time ago as a few static HTML pages. Over the years it migrated through a couple of
different approaches before settling on [Jekyll](https://jekyllrb.com) and [GitHub Pages](https://pages.github.com),
which is what you're looking at now. The portfolio, journey and notes sections are all written in Markdown and rendered
by Jekyll using the [Minima](https://github.com/jekyll/minima) theme.

Feel free to fork it, borrow bits of it or just have a look around.

## Local development

You'll need Ruby and Jekyll. If you're on macOS and starting from scratch:

```bash
brew install rbenv
echo 'eval "$(rbenv init -)"' >> ~/.zshrc
source ~/.zshrc
rbenv install 3.3.10
rbenv global 3.3.10
ruby -v
gem install jekyll bundler
bundle install
```

Then to run the site locally with live reload:

```bash
bundle exec jekyll serve --livereload
```

The site will be available at `http://localhost:4000`.
