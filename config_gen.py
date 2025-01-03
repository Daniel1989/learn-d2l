import json
import gitlab

GITLAB_URL = 'https://git.ywwl.com'
ACCESS_TOKEN = 'HDV_nPGpmBG2fkQST7Th'
gl = gitlab.Gitlab(GITLAB_URL, private_token=ACCESS_TOKEN)
gl.auth()
projects = gl.projects.list(all=True)
repos = {
}
for item in projects:
    if item.namespace["name"] in ['YWFE', 'ywCloud', 'ywfe']:
        branch = None
        try:
            item.branches.get("release")
            branch = "release"
        except gitlab.exceptions.GitlabGetError:
            try:
                item.branches.get("master")
                branch = "master"
            except gitlab.exceptions.GitlabGetError:
                print(item.path_with_namespace + "不存在master或release分支")
        if branch is not None:
            repos[item.path_with_namespace + "("+branch+")"] = {
                "url": item.ssh_url_to_repo,
                "vcs-config": {
                    "ref": branch
                }
            }


with open("./config.json", 'w') as outfile:
    settings = {
            "dbpath": "db",
            "repos": repos
        }
    json.dump(settings, outfile, indent=4, sort_keys=True)
    outfile.write('\n')

